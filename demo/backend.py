"""NIfTI viewing and full-checkpoint inference for the RevCT demo."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import spaces
from PIL import Image

MODEL_ID = "hcltech-robotics/rev-ct"
MODEL_REVISION = "v0.3.2"
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
MAX_VOXELS = 256 * 1024 * 1024
WINDOWS = {"Lung": (-1350, 150), "Mediastinum": (-160, 240), "Bone": (-450, 1050)}
PLANES = {"Axial": 2, "Coronal": 1, "Sagittal": 0}


@dataclass(frozen=True)
class VolumePreview:
    data: np.ndarray
    spacing: tuple[float, float, float]
    original_shape: tuple[int, int, int]


def validate_volume(file_path: str) -> tuple[Path, nib.spatialimages.SpatialImage]:
    if not file_path:
        raise ValueError("Upload a chest CT in NIfTI format first.")
    path = Path(file_path)
    if not path.name.lower().endswith((".nii", ".nii.gz")):
        raise ValueError("Choose a .nii or .nii.gz CT volume.")
    if not path.is_file():
        raise ValueError("The uploaded file is no longer available. Please upload it again.")
    if path.stat().st_size > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a CT volume smaller than 512 MB.")
    try:
        image = nib.load(path)
    except (OSError, ValueError, nib.filebasedimages.ImageFileError) as error:
        raise ValueError("This file could not be read as a NIfTI volume.") from error
    if len(image.shape) != 3 or any(size < 2 for size in image.shape):
        raise ValueError("Upload one three-dimensional CT volume, rather than a 2D image or 4D series.")
    if np.prod(image.shape, dtype=np.int64) > MAX_VOXELS:
        raise ValueError("This volume is too large for the shared demo. Use the local inference package.")
    affine = image.affine
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
        raise ValueError("The volume needs valid physical orientation and voxel spacing.")
    if not all(np.isfinite(v) and v > 0 for v in image.header.get_zooms()[:3]):
        raise ValueError("The volume needs positive physical voxel spacing.")
    return path, image


@lru_cache(maxsize=2)
def _read_preview(file_path: str, modified_ns: int, time_bucket: int) -> VolumePreview:
    _, image = validate_volume(file_path)
    steps = np.maximum(1, np.ceil(np.asarray(image.shape) / 256).astype(int))
    try:
        # Read a bounded overview, retaining the physical axes for display.
        selection = tuple(slice(None, None, int(step)) for step in steps)
        data = np.asarray(image.dataobj[selection], dtype=np.float32)
    except (OSError, ValueError, EOFError) as error:
        raise ValueError("The CT volume is incomplete or damaged. Please choose another file.") from error
    if not np.isfinite(data).all():
        raise ValueError("The CT contains invalid intensity values.")
    affine = image.affine.copy()
    affine[:3, :3] *= steps
    overview = nib.as_closest_canonical(nib.Nifti1Image(data, affine))
    return VolumePreview(
        np.asarray(overview.dataobj, dtype=np.float32),
        tuple(float(v) for v in overview.header.get_zooms()[:3]),
        tuple(int(v) for v in image.shape),
    )


class DemoBackend:
    def __init__(self, model_id: str | None = None, revision: str | None = None):
        self.model_id = model_id or os.environ.get("REV_MODEL_ID", MODEL_ID)
        self.revision = revision or os.environ.get("REV_MODEL_REVISION", MODEL_REVISION)
        self._model = None
        self._processor = None
        self._lock = threading.Lock()
        self._zero_gpu = os.environ.get("SPACES_ZERO_GPU", "").lower() in {"1", "t", "true"}
        # Wrap the bound function so only CPU tensors and the volume name enter
        # the worker queue; the preloaded model is inherited by its GPU worker.
        self._forward = spaces.GPU(self._forward, duration=30)

    def prepare_for_host(self) -> None:
        if self._zero_gpu:
            # ZeroGPU captures CUDA placements during startup for fast allocation.
            self._load_model()

    def available_examples(self) -> list[list[str]]:
        directory = Path(__file__).parent / "examples"
        return [[str(path)] for path in sorted(directory.glob("*.nii.gz"))]

    def describe(self, file_path: str) -> str:
        _, image = validate_volume(file_path)
        shape = " × ".join(str(v) for v in image.shape)
        spacing = " × ".join(f"{v:.2f}" for v in image.header.get_zooms()[:3])
        return f"{shape} voxels · {spacing} mm"

    def preview(self, file_path: str, plane: str, position: float, window: str) -> np.ndarray:
        path, _ = validate_volume(file_path)
        if plane not in PLANES or window not in WINDOWS:
            raise ValueError("Choose one of the available viewing planes and CT windows.")
        if not np.isfinite(position) or not 0 <= position <= 1:
            raise ValueError("Slice position must be between 0 and 1.")
        volume = _read_preview(str(path), path.stat().st_mtime_ns, int(time.monotonic() // 300))
        axis = PLANES[plane]
        index = int(round(position * (volume.data.shape[axis] - 1)))
        pixels = np.take(volume.data, index, axis=axis)
        low, high = WINDOWS[window]
        pixels = (np.clip((pixels - low) / (high - low), 0, 1) * 255).astype(np.uint8)
        # RAS overview; frontal planes are displayed with superior at the top.
        if plane == "Axial":
            pixels = np.flip(pixels.T, axis=(0, 1))
            spacing = (volume.spacing[0], volume.spacing[1])
        elif plane == "Coronal":
            pixels = np.flip(pixels.T, axis=(0, 1))
            spacing = (volume.spacing[0], volume.spacing[2])
        else:
            pixels = np.flipud(pixels.T)
            spacing = (volume.spacing[1], volume.spacing[2])
        width, height = pixels.shape[1] * spacing[0], pixels.shape[0] * spacing[1]
        scale = 640 / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return np.asarray(Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR))

    def _load_model(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoProcessor

        import rev_ct  # noqa: F401 -- registers the Transformers Auto classes

        torch.set_num_threads(int(os.environ.get("REV_NUM_THREADS", "2")))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = "cuda" if self._zero_gpu else (
            os.environ.get("REV_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = AutoModel.from_pretrained(
            self.model_id, revision=self.revision, trust_remote_code=False,
        ).eval().to(device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, revision=self.revision, preprocessing="fast", trust_remote_code=False,
        )

    def _forward(self, inputs: dict, volume_name: str) -> dict:
        import torch

        with torch.inference_mode():
            return self._model(**inputs, volume_names=[volume_name]).records[0]

    def _cached_example(self, path: Path) -> dict | None:
        directory = Path(__file__).parent / "examples"
        metadata_path = directory / "lung1.provenance.json"
        if (
            self.model_id != MODEL_ID or self.revision != MODEL_REVISION
            or os.environ.get("REV_DISABLE_EXAMPLE_CACHE") == "1"
            or not metadata_path.is_file()
        ):
            return None
        metadata = json.loads(metadata_path.read_text())
        if path.stat().st_size != metadata["input_bytes"]:
            return None
        with path.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        if digest != metadata["input_sha256"]:
            return None
        record = json.loads((directory / "lung1.result.json").read_text())
        record["volume"] = path.name
        return record

    def infer(self, file_path: str) -> tuple[dict, float | None]:
        path, _ = validate_volume(file_path)
        # Validate the compressed payload and finite intensities before preprocessing.
        _read_preview(str(path), path.stat().st_mtime_ns, int(time.monotonic() // 300))
        cached = self._cached_example(path)
        if cached is not None:
            return cached, None
        import torch

        with self._lock:
            self._load_model()
            start = time.perf_counter()
            inputs = self._processor(images3d=[str(path)], return_tensors="pt")
            if not torch.isfinite(inputs["pixel_values"]).all():
                raise ValueError("The CT contains invalid intensity values.")
            record = self._forward(inputs, path.name)
            elapsed = time.perf_counter() - start
        return record, elapsed
