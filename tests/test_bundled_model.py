import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from colipri.checkpoint import load_model_config
from omegaconf import OmegaConf
from safetensors.numpy import load_file, save_file
from safetensors.torch import load_file as load_torch_file

from rev_ct.configuration_rev_ct import RevConfig
from rev_ct.head import RevHead
from rev_ct.modeling_rev_ct import RevModel

ROOT = Path(__file__).resolve().parents[1]


def image_model_config():
    config = load_model_config()
    config.text_encoder = None
    return OmegaConf.to_container(config, resolve=True)


def test_bundled_encoder_is_registered_without_a_weight_download(monkeypatch):
    def no_download(*args, **kwargs):
        raise AssertionError("A bundled model must not download another checkpoint")

    monkeypatch.setattr("colipri.model.multimodal.download_weights", no_download)
    config = json.loads((ROOT / "artifacts/config.json").read_text())
    config.update(encoder_bundled=True, encoder_model_config=image_model_config())
    with init_empty_weights():
        model = RevModel(RevConfig(**config))
    assert "encoder" in dict(model.named_children())
    assert any(name.startswith("encoder.image_encoder.") for name in model.state_dict())
    assert model.encoder.text_encoder is None
    assert model.encoder.softmax_temperature.dtype == torch.float32
    assert model.dtype == torch.float32
    assert not model.encoder.training
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    model.load_state_dict(load_torch_file(ROOT / "artifacts/model.safetensors"), strict=False)
    sample = np.load(ROOT / "artifacts/validation_sample.npz")
    output = model(features=torch.from_numpy(sample["features"]))
    np.testing.assert_allclose(output.probabilities.numpy(), sample["scores"], atol=1e-7)
    np.testing.assert_array_equal(output.decisions.numpy(), sample["decisions"])


def test_feature_only_loading_skips_bundled_encoder_tensors(tmp_path):
    config = json.loads((ROOT / "artifacts/config.json").read_text())
    config["encoder_bundled"] = True
    (tmp_path / "config.json").write_text(json.dumps(config))
    tensors = load_file(ROOT / "artifacts/model.safetensors")
    tensors["encoder.image_encoder.unused"] = np.ones((2, 3), dtype=np.float32)
    save_file(tensors, tmp_path / "model.safetensors")
    head = RevHead.from_pretrained(tmp_path)
    assert not any(name.startswith("encoder.") for name in head.tensors)
    assert head.encoder_checkpoint == tmp_path / "model.safetensors"
    sample = np.load(ROOT / "artifacts/validation_sample.npz")
    np.testing.assert_array_equal(head.predict(sample["features"]).decisions, sample["decisions"])


def test_transformers_reinitialises_the_nonpersistent_rotary_cache():
    config = json.loads((ROOT / "artifacts/config.json").read_text())
    config.update(encoder_bundled=True, encoder_model_config=image_model_config())
    with init_empty_weights():
        model = RevModel(RevConfig(**config))
    rope = model.encoder.image_encoder.backbone.eva.rope
    expected = rope.pos_embed.clone()
    # Transformers allocates this non-checkpoint buffer without initialising it.
    rope.pos_embed.fill_(float("nan"))
    model._init_weights(rope)
    torch.testing.assert_close(rope.pos_embed, expected, atol=0, rtol=0)


def test_bundle_export_rejects_a_checkpoint_without_the_image_encoder(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_release.py"),
            str(tmp_path / "release"),
            "--bundle-encoder",
            "--encoder-checkpoint",
            str(ROOT / "artifacts/model.safetensors"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "Checkpoint does not contain the complete COLIPRI image encoder" in result.stderr
