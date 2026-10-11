import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import validate

from rev_ct.head import RevHead

ROOT = Path(__file__).resolve().parents[1]


def test_real_validation_sample_matches_original_scores_and_decisions():
    head = RevHead.from_pretrained(ROOT / "artifacts")
    data = np.load(ROOT / "artifacts/validation_sample.npz")
    out = head.predict(data["features"])
    # BLAS float32 reductions differ between a full 3039-row batch and a 32-row batch.
    np.testing.assert_allclose(out.raw_probabilities, data["raw_probabilities"], atol=5e-6, rtol=0)
    np.testing.assert_allclose(out.scores, data["scores"], atol=1e-7)
    np.testing.assert_array_equal(out.decisions, data["decisions"])
    schema = json.loads((ROOT / "artifacts/typed_schema.json").read_text())
    for record in head.records(data["features"], data["names"]):
        validate(record, schema)
        assert len(record["decisions"]) == 18


@pytest.mark.parametrize(
    "x",
    [
        np.zeros((2, 767)),
        np.zeros((1, 1, 768)),
        np.full((1, 768), np.nan),
        np.full((1, 768), np.inf),
    ],
)
def test_bad_features_fail_before_inference(x):
    head = RevHead.from_pretrained(ROOT / "artifacts")
    with pytest.raises(ValueError):
        head.predict(x)


def test_isotonic_clips_extreme_probabilities():
    head = RevHead.from_pretrained(ROOT / "artifacts")
    low = head.calibrate(np.zeros((1, 18)))
    high = head.calibrate(np.ones((1, 18)))
    assert np.all(low <= high)
    assert np.all((low >= 0) & (high <= 1))


def test_empty_feature_batch_returns_no_records():
    head = RevHead.from_pretrained(ROOT / "artifacts")
    assert head.records(np.empty((0, 768)), []) == []


def test_names_must_match_batch_and_mode_must_be_known():
    head = RevHead.from_pretrained(ROOT / "artifacts")
    with pytest.raises(ValueError):
        head.records(np.zeros((2, 768)), ["one"])
    with pytest.raises(ValueError):
        head.predict(np.zeros((1, 768)), mode="wrong")


def test_config_rejects_duplicate_or_incomplete_labels(tmp_path):
    cfg = json.loads((ROOT / "artifacts/config.json").read_text())
    cfg["labels"] = ["Medical material"] * 18
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="labels"):
        RevHead.from_pretrained(tmp_path)
