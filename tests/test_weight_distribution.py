"""Installed inference must work with Hub checkpoints containing no Python code."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoProcessor

import rev_ct  # noqa: F401 — importing the installed package registers Transformers classes.

ROOT = Path(__file__).resolve().parents[1]


def weight_only_fixture(path):
    config = json.loads((ROOT / "artifacts/config.json").read_text())
    config["model_type"] = "decision-model"
    config.pop("auto_map", None)
    (path / "config.json").write_text(json.dumps(config))
    (path / "processor_config.json").write_text(json.dumps({"processor_class": "RevProcessor"}))
    shutil.copyfile(ROOT / "artifacts/model.safetensors", path / "model.safetensors")


def test_installed_auto_classes_load_a_checkpoint_without_remote_code(tmp_path):
    weight_only_fixture(tmp_path)
    assert AutoConfig.from_pretrained(tmp_path).model_type == "decision-model"
    model = AutoModel.from_pretrained(tmp_path, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    sample = np.load(ROOT / "artifacts/validation_sample.npz")
    output = model(features=torch.from_numpy(sample["features"]))
    np.testing.assert_allclose(output.probabilities.numpy(), sample["scores"], atol=1e-7)
    np.testing.assert_array_equal(output.decisions.numpy(), sample["decisions"])
    assert processor.model_input_names == ["pixel_values"]


def test_saved_checkpoint_needs_only_the_installed_package(tmp_path):
    weight_only_fixture(tmp_path)
    model = AutoModel.from_pretrained(tmp_path, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=False)
    saved = tmp_path / "saved"
    model.save_pretrained(saved)
    processor.save_pretrained(saved)
    assert not list(saved.rglob("*.py"))
    assert "auto_map" not in json.loads((saved / "config.json").read_text())
    restored = AutoModel.from_pretrained(saved, trust_remote_code=False)
    assert AutoProcessor.from_pretrained(saved).preprocessing == "reference"
    features = torch.zeros(2, 768)
    torch.testing.assert_close(restored(features=features).probabilities, model(features=features).probabilities)


def test_hub_export_contains_weights_and_metadata_without_repository_source(tmp_path):
    spec = importlib.util.spec_from_file_location("build_release", ROOT / "scripts/build_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    destination = module.build_release(tmp_path / "hub")
    assert not list(destination.rglob("*.py"))
    assert not (destination / "pyproject.toml").exists()
    assert not (destination / "tests").exists()
    model = AutoModel.from_pretrained(destination, trust_remote_code=False)
    assert model.config.model_type == "decision-model"


def test_installed_package_loads_legacy_metadata_without_remote_code():
    model = AutoModel.from_pretrained(ROOT / "artifacts", trust_remote_code=False)
    assert model.config.encoder_bundled is False
    features = np.load(ROOT / "artifacts/validation_sample.npz")
    np.testing.assert_array_equal(
        model(features=torch.from_numpy(features["features"])).decisions.numpy(), features["decisions"]
    )


def test_uploader_rejects_source_files_before_contacting_the_hub(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("upload_release", ROOT / "scripts/upload_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for filename in ["model.safetensors", "config.json", "processor_config.json", "README.md", "LICENSE"]:
        (tmp_path / filename).touch()
    (tmp_path / "inference.py").write_text("# Code belongs on GitHub")
    monkeypatch.setattr(module, "HfApi", lambda: pytest.fail("Invalid exports must be rejected before contacting the Hub"))
    with pytest.raises(ValueError, match="source"):
        module.upload_release(tmp_path, "chrisvoncsefalvay/rev-ct")


def test_uploader_cannot_replace_the_complete_model_with_head_only_fixtures(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("upload_release", ROOT / "scripts/upload_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for filename in ["model.safetensors", "config.json", "processor_config.json"]:
        shutil.copyfile(ROOT / "artifacts" / filename, tmp_path / filename)
    for filename in ["README.md", "LICENSE"]:
        shutil.copyfile(ROOT / filename, tmp_path / filename)
    monkeypatch.setattr(module, "HfApi", lambda: pytest.fail("Incomplete checkpoints must not contact the Hub"))
    with pytest.raises(ValueError, match="complete"):
        module.upload_release(tmp_path, "chrisvoncsefalvay/rev-ct")
