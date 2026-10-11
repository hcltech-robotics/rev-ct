from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoModel, AutoProcessor

from rev_ct.release import systemone

ROOT = Path(__file__).resolve().parents[1]


def test_auto_model_loads_safetensors_and_returns_typed_output(tmp_path):
    model = AutoModel.from_pretrained(ROOT / "artifacts", trust_remote_code=True)
    z = np.load(ROOT / "artifacts/validation_sample.npz")
    with torch.inference_mode():
        out = model(features=torch.from_numpy(z["features"]), volume_names=z["names"].tolist())
    assert out.logits.shape == (32, 18)
    np.testing.assert_allclose(out.probabilities.numpy(), z["scores"], atol=1e-7)
    np.testing.assert_array_equal(out.decisions.numpy(), z["decisions"])
    assert len(out.records) == 32
    model.save_pretrained(tmp_path)
    reloaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True)
    np.testing.assert_array_equal(
        reloaded(features=torch.from_numpy(z["features"])).decisions, out.decisions
    )


def test_auto_processor_loads_without_downloading_an_encoder():
    processor = AutoProcessor.from_pretrained(ROOT / "artifacts", trust_remote_code=True)
    assert processor.model_input_names == ["pixel_values"]


def test_clef_style_systemone_selects_a_trained_noul_question():
    model = AutoModel.from_pretrained(ROOT / "artifacts", trust_remote_code=True)
    z = np.load(ROOT / "artifacts/validation_sample.npz")
    response = systemone(
        model,
        None,
        {
            "model": "rev-ct",
            "state": {"features": z["features"][0].tolist()},
            "questions": {"effusion": {"type": "noul", "finding": "Pleural effusion"}},
        },
    )
    assert set(response["answers"]) == {"effusion"}
    assert response["answers"]["effusion"]["type"] == "noul"
    assert response["answers"]["effusion"]["noul"] == round(float(z["scores"][0, 12]), 4)
    assert response["usage"]["output_tokens"] == 0


def test_systemone_rejects_unsupported_questions_before_encoding():
    model = AutoModel.from_pretrained(ROOT / "artifacts", trust_remote_code=True)
    with pytest.raises(ValueError, match="finding"):
        systemone(
            model, None, {"state": {}, "questions": {"q": {"type": "noul", "finding": "Stroke"}}}
        )


def test_forward_computes_head_predictions_once(monkeypatch):
    model = AutoModel.from_pretrained(ROOT / "artifacts", trust_remote_code=True)
    head_class = type(model._rev_head())
    original = head_class.predict
    calls = []

    def counted(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(head_class, "predict", counted)
    z = np.load(ROOT / "artifacts/validation_sample.npz")
    out = model(features=torch.from_numpy(z["features"]))
    np.testing.assert_allclose(out.probabilities.numpy(), z["scores"], atol=1e-7)
    assert len(calls) == 1
