"""Synchronised CUDA benchmark utilities; no model downloads occur inside timers."""

from __future__ import annotations

import numpy as np


def summarise(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("No timing samples")
    times = np.array([sample["total_s"] for sample in samples], dtype=float)
    if np.any(~np.isfinite(times)) or np.any(times <= 0):
        raise ValueError("Timing samples must be finite and positive")
    return {
        "n": len(times),
        "mean_s": float(times.mean()),
        "p50_s": float(np.median(times)),
        "p95_s": float(np.percentile(times, 95)),
        "volumes_per_second": float(len(times) / times.sum()),
    }
