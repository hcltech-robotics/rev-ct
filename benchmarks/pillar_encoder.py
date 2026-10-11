"""Pillar-0 encoder timing at its released shape, excluding CT preprocessing."""

import importlib.metadata
import json
import time

import numpy as np
import torch
from transformers import AutoModel

REPO = "YalaLab/Pillar0-ChestCT"
REVISION = "b9dfd833a947744382d4f1bd73b5dd72fcb58b34"


def main():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True).eval().cuda()
    torch.cuda.synchronize()
    result = {
        "repo": REPO,
        "revision": REVISION,
        "gpu": torch.cuda.get_device_name(),
        "input_shape": [1, 11, 256, 256, 256],
        "input": "synthetic seeded tensor",
        "scope": "vision encoder only; excludes official 11-window CT preprocessing and finding heads",
        "load_s": time.perf_counter() - started,
        "parameters": sum(p.numel() for p in model.parameters()),
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "timm"]},
        "results": {},
    }
    pixels = torch.randn(*result["input_shape"], device="cuda")
    for precision, dtype in [("fp32", torch.float32), ("amp_bf16", torch.bfloat16)]:
        inputs = pixels.to(dtype=dtype)
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)),
        ):
            output = model.extract_vision_feats(image={"chest_ct": inputs})
        torch.cuda.synchronize()
        first_s = time.perf_counter() - started
        if tuple(output.shape) != (1, 1152) or not torch.isfinite(output).all():
            raise ValueError("Invalid Pillar vision output")
        times = []
        for _ in range(16):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with (
                torch.inference_mode(),
                torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16)),
            ):
                model.extract_vision_feats(image={"chest_ct": inputs})
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        result["results"][precision] = {
            "n": len(times),
            "first_forward_s": first_s,
            "p50_encoder_s": float(np.median(times)),
            "p95_encoder_s": float(np.percentile(times, 95)),
            "mean_encoder_s": float(np.mean(times)),
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "samples_s": times,
        }
        print("PILLAR_RESULT " + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
