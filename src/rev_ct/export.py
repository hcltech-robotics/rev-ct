"""One-time conversion of a trusted training artifact to tensor-only inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ENCODER_REVISION = "892e42ec363f6c561b89c459d60fcd95ac1a99bf"
SOURCE_REPO = "chrisvoncsefalvay/ct-rate-colipri-probe"


def export_artifact(
    artifact: dict,
    destination: str | Path,
    *,
    source_revision: str,
    encoder_revision: str = ENCODER_REVISION,
) -> None:
    from .head import RevHead

    clfs = artifact["clfs"]
    if len(clfs) != 18 or any(kind != "lr" for kind, _ in clfs):
        raise ValueError("Expected 18 logistic regression heads")
    if any(not np.array_equal(clf.classes_, [0, 1]) for _, clf in clfs):
        raise ValueError("Heads must use ordered binary classes [0, 1]")
    scaler = artifact["scaler"]
    weight = np.stack([clf.coef_[0] for _, clf in clfs])
    bias = np.array([clf.intercept_[0] for _, clf in clfs])
    folded = weight / scaler.scale_
    raw_bias = bias - folded @ scaler.mean_
    tensors = {
        "head.original_weight": weight,
        "head.original_bias": bias,
        "scaler.mean": scaler.mean_,
        "scaler.scale": scaler.scale_,
        "head.weight": folded,
        "head.raw_bias": raw_bias,
        "head.decision_bias": raw_bias - artifact["biases"].astype(np.float64),
        "decision.threshold_bias": artifact["biases"],
    }
    if len(artifact["isotonic"]) != 18:
        raise ValueError("Expected 18 isotonic maps")
    for j, calibrator in enumerate(artifact["isotonic"]):
        tensors[f"calibration.{j}.x"] = calibrator.X_thresholds_
        tensors[f"calibration.{j}.y"] = calibrator.y_thresholds_
    config = {
        "format_version": "rev-safetensors-1",
        "schema_version": "rev-1.0",
        "labels": list(artifact["label_cols"]),
        "feature_dim": 768,
        "calibration_input": "raw_probability",
        "reference_dtype": str(weight.dtype),
        "folded_dtype": str(folded.dtype),
        "default_head_mode": "reference",
        "encoder": {
            "repo_id": "microsoft/colipri",
            "revision": encoder_revision,
            "package_version": "0.1.3",
            "pool": True,
            "project": True,
            "normalize": False,
            "input_size": [192, 192, 192],
            "dtype": "float32",
            "feature_storage_dtype": "float16",
        },
        "source": {
            "repo_id": SOURCE_REPO,
            "revision": source_revision,
            "artifact": "typed_calibration.joblib",
        },
    }
    RevHead(config, tensors)  # Refuse invalid artifacts before writing a release.
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    save_file(
        {k: np.ascontiguousarray(v) for k, v in tensors.items()},
        destination / "model.safetensors",
        metadata={"format": "rev-safetensors-1"},
    )
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
