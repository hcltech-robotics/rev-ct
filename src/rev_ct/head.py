"""Safe, CPU-only inference for the Rev linear heads and isotonic calibration."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from safetensors import safe_open
from scipy.special import expit


@dataclass(frozen=True)
class HeadOutput:
    raw_probabilities: np.ndarray
    scores: np.ndarray
    decisions: np.ndarray
    adjusted_logits: np.ndarray


def sigmoid(x: np.ndarray) -> np.ndarray:
    return expit(x)


class RevHead:
    def __init__(self, config: dict, tensors: dict[str, np.ndarray]):
        self.config = config
        self.labels = tuple(config["labels"])
        if len(self.labels) != 18 or len(set(self.labels)) != 18:
            raise ValueError("Rev requires 18 distinct labels in source order")
        if config.get("format_version") != "rev-safetensors-1":
            raise ValueError("Unsupported Rev artifact format")
        if config.get("schema_version") != "rev-1.0" or config.get("feature_dim") != 768:
            raise ValueError("Unsupported Rev schema or feature dimension")
        if config.get("calibration_input") != "raw_probability":
            raise ValueError("Rev calibration requires raw_probability")
        self.tensors = tensors
        self.encoder_checkpoint = None
        shapes = {
            "head.original_weight": (18, 768),
            "head.original_bias": (18,),
            "scaler.mean": (768,),
            "scaler.scale": (768,),
            "head.weight": (18, 768),
            "head.raw_bias": (18,),
            "head.decision_bias": (18,),
            "decision.threshold_bias": (18,),
        }
        for key, shape in shapes.items():
            if key not in tensors or tensors[key].shape != shape:
                raise ValueError(f"Invalid tensor {key}: expected {shape}")
            if not np.isfinite(tensors[key]).all():
                raise ValueError(f"Nonfinite tensor {key}")
        if np.any(tensors["scaler.scale"] <= 0):
            raise ValueError("Scaler scale must be positive")
        for j in range(18):
            x, y = tensors[f"calibration.{j}.x"], tensors[f"calibration.{j}.y"]
            if x.ndim != 1 or x.shape != y.shape or not len(x):
                raise ValueError(f"Invalid isotonic table for label {j}")
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"Nonfinite isotonic table for label {j}")
            if np.any(np.diff(x) <= 0) or np.any(np.diff(y) < 0) or np.any((y < 0) | (y > 1)):
                raise ValueError(f"Invalid isotonic bounds/order for label {j}")

    @classmethod
    def from_pretrained(
        cls, source: str | Path, *, revision: str | None = None, local_files_only: bool = False
    ) -> RevHead:
        source = str(source)
        if Path(source).is_dir():
            root = Path(source)
        else:
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    source,
                    revision=revision,
                    allow_patterns=["config.json", "model.safetensors"],
                    local_files_only=local_files_only,
                )
            )
        config = json.loads((root / "config.json").read_text())
        # Validate config before opening weights, so malformed labels fail clearly.
        if len(config.get("labels", [])) != 18 or len(set(config["labels"])) != 18:
            raise ValueError("Rev requires 18 distinct labels in source order")
        checkpoint = root / "model.safetensors"
        # Feature-only inference must not materialise the bundled image encoder.
        with safe_open(checkpoint, framework="np") as weights:
            tensors = {
                name: weights.get_tensor(name)
                for name in weights.keys()
                if not name.startswith("encoder.")
            }
        head = cls(config, tensors)
        if config.get("encoder_bundled"):
            head.encoder_checkpoint = checkpoint
        return head

    def _features(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 1:
            x = x[None, :]
        if x.ndim != 2 or x.shape[1] != 768 or not np.isfinite(x).all():
            raise ValueError("Expected finite features with shape (N, 768) or (768,)")
        return x

    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        p = np.asarray(probabilities)
        if p.ndim != 2 or p.shape[1] != 18 or not np.isfinite(p).all():
            raise ValueError("Expected finite probabilities with shape (N, 18)")
        if np.any((p < 0) | (p > 1)):
            raise ValueError("Probabilities must be in [0, 1]")
        out = np.empty_like(p, dtype=np.float32)
        for j in range(18):
            x, y = self.tensors[f"calibration.{j}.x"], self.tensors[f"calibration.{j}.y"]
            # sklearn casts input to X_thresholds_' dtype before interpolation.
            out[:, j] = np.interp(p[:, j].astype(x.dtype), x, y).astype(y.dtype)
        return out

    def predict(self, features: np.ndarray, *, mode: str = "reference") -> HeadOutput:
        x = self._features(features)
        t = self.tensors
        if mode == "reference":
            z = x.copy()
            z -= t["scaler.mean"].astype(z.dtype)
            z /= t["scaler.scale"].astype(z.dtype)
            raw = np.column_stack(
                [
                    (
                        z @ t["head.original_weight"][j : j + 1].T + t["head.original_bias"][j]
                    ).ravel()
                    for j in range(18)
                ]
            )
            adjusted = raw - t["decision.threshold_bias"]
        elif mode == "fused":
            raw = x.astype(np.float64) @ t["head.weight"].T + t["head.raw_bias"]
            adjusted = raw - t["decision.threshold_bias"]
        else:
            raise ValueError("mode must be reference or fused")
        p = sigmoid(raw)
        return HeadOutput(p, self.calibrate(p), adjusted >= 0, adjusted)

    def records(
        self, features: np.ndarray, names: Sequence[str], *, mode: str = "reference"
    ) -> list[dict]:
        out = self.predict(features, mode=mode)
        return self.records_from_output(out, names)

    def records_from_output(self, out: HeadOutput, names: Sequence[str]) -> list[dict]:
        """Format an existing prediction without running the heads again."""
        if len(names) != len(out.scores):
            raise ValueError("Volume names must match the feature batch")
        records = []
        for name, scores, decisions in zip(names, out.scores, out.decisions, strict=True):
            positives = [j for j in range(18) if decisions[j]]
            top = sorted(positives, key=lambda j: (-float(scores[j]), j))[:3]
            records.append(
                {
                    "volume": str(name),
                    "schema_version": "rev-1.0",
                    "decisions": [
                        {
                            "finding": label,
                            "decision": "present" if bool(present) else "absent",
                            "confidence": float(score) if present else 1.0 - float(score),
                            "score": float(score),
                        }
                        for label, score, present in zip(
                            self.labels, scores, decisions, strict=True
                        )
                    ],
                    "choice": {"top_findings": [self.labels[j] for j in top]},
                    "score": {"burden": float(scores.mean()), "n_positive": len(positives)},
                }
            )
        return records
