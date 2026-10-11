"""NIfTI preprocessing and optional CUDA COLIPRI inference."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .head import RevHead

_worker_processor = None


def create_image_encoder(config: dict):
    """Construct the checkpoint's image architecture without fetching weights."""
    from colipri import get_model
    from omegaconf import OmegaConf

    architecture = config.get("encoder_model_config")
    if not architecture or architecture.get("text_encoder") is not None:
        raise ValueError("Bundled Rev requires an image-only encoder_model_config")
    architecture = dict(architecture)
    # COLIPRI's default literal is an integer; Transformers retains buffer dtype.
    architecture["temperature"] = float(architecture.get("temperature", 1.0))
    return get_model(config=OmegaConf.create(architecture), pretrained=False, image_only=True)


def load_bundled_encoder(head: RevHead):
    from safetensors import safe_open

    if head.encoder_checkpoint is None:
        raise ValueError("Load bundled Rev with RevHead.from_pretrained to locate its checkpoint")
    model = create_image_encoder(head.config)
    with safe_open(head.encoder_checkpoint, framework="pt", device="cpu") as weights:
        state = {
            name.removeprefix("encoder."): weights.get_tensor(name)
            for name in weights.keys()
            if name.startswith("encoder.")
        }
    model.load_state_dict(state, strict=True)
    return model


def encode_features(model, pixel_values, device, dtype) -> np.ndarray:
    """Preserve the historical fp16 feature-storage rounding in either layout."""
    import torch

    if pixel_values.ndim != 5 or tuple(pixel_values.shape[1:]) != (1, 192, 192, 192):
        raise ValueError("Expected preprocessed images with shape (N, 1, 192, 192, 192)")
    with torch.inference_mode():
        inputs = pixel_values.to(device=device, dtype=dtype, non_blocking=True)
        features = model.encode_image(inputs, pool=True, project=True, normalize=False)
        return features.detach().to(device="cpu", dtype=torch.float16).numpy().astype(np.float32)


def _init_worker(preprocessing: str = "reference") -> None:
    global _worker_processor
    import SimpleITK as sitk
    import torch
    from colipri import get_processor

    torch.set_num_threads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    if preprocessing == "fast":
        from .preprocessing import FastImageProcessor

        _worker_processor = FastImageProcessor()
    else:
        _worker_processor = get_processor(image_only=True)


def _process_path(path: str):
    return _worker_processor.to_images_batch(_worker_processor.process_images(path))[0]


def preprocess_paths(paths: Iterable[str], workers: int, preprocessing: str = "reference"):
    """Bound outstanding preprocessing tensors; spawn never inherits CUDA state."""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
        initializer=_init_worker,
        initargs=(preprocessing,),
    ) as pool:
        pending = deque()
        iterator = iter(paths)
        for _ in range(workers * 2):
            path = next(iterator, None)
            if path is None:
                break
            pending.append((path, pool.submit(_process_path, path)))
        while pending:
            name, future = pending.popleft()
            yield name, future.result()
            path = next(iterator, None)
            if path is not None:
                pending.append((path, pool.submit(_process_path, path)))


class RevInference:
    def __init__(
        self,
        head: RevHead,
        *,
        device: str = "cuda",
        precision: str = "fp32",
        compile_mode: str | None = None,
        preprocessing: str = "reference",
    ):
        import torch
        from colipri import get_model, get_processor
        from huggingface_hub import hf_hub_download

        if precision not in ("bf16", "fp32"):
            raise ValueError("precision must be bf16 or fp32")
        if preprocessing not in ("reference", "fast"):
            raise ValueError("preprocessing must be reference or fast")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; choose device='cpu', precision='fp32'")
        if precision == "bf16" and device.startswith("cuda") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bf16")
        self.head = head
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        encoder = head.config["encoder"]
        if head.config.get("encoder_bundled"):
            self.model = load_bundled_encoder(head).eval()
        else:
            checkpoint = hf_hub_download(
                encoder["repo_id"], "model.safetensors", revision=encoder["revision"]
            )
            self.model = get_model(checkpoint_path=checkpoint, image_only=True).eval()
        self.model.to(device=self.device, dtype=self.dtype)
        self.preprocessing = preprocessing
        if preprocessing == "fast":
            from .preprocessing import FastImageProcessor

            self.processor = FastImageProcessor()
        else:
            self.processor = get_processor(image_only=True)
        if compile_mode is not None:
            self.model.image_encoder = torch.compile(self.model.image_encoder, mode=compile_mode)

    def preprocess(self, paths: list[str]):
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        return self.processor.to_images_batch(self.processor.process_images(paths))

    def encode(self, pixel_values) -> np.ndarray:
        return encode_features(self.model, pixel_values, self.device, self.dtype)

    def predict(
        self,
        paths: list[str],
        *,
        workers: int = 0,
        batch_size: int = 1,
        head_mode: str = "reference",
    ):
        import torch

        if workers < 0 or batch_size < 1:
            raise ValueError("workers must be >= 0 and batch_size must be >= 1")
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        if not workers:
            for i in range(0, len(paths), batch_size):
                chunk = paths[i : i + batch_size]
                yield from self.head.records(
                    self.encode(self.preprocess(chunk)), chunk, mode=head_mode
                )
            return
        names, tensors = [], []
        for name, tensor in preprocess_paths(paths, workers, self.preprocessing):
            names.append(name)
            tensors.append(tensor)
            if len(names) == batch_size:
                yield from self.head.records(
                    self.encode(torch.stack(tensors)), names, mode=head_mode
                )
                names, tensors = [], []
        if names:
            yield from self.head.records(self.encode(torch.stack(tensors)), names, mode=head_mode)
