"""Build a self-contained Hub release from the audited tensor artifacts."""

import argparse
import json
import shutil
from pathlib import Path

from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]


def bundle_encoder(destination, checkpoint_path=None):
    """Combine the pinned image-only COLIPRI weights and unchanged Rev tensors."""
    from accelerate import init_empty_weights
    from colipri import get_model
    from colipri.checkpoint import load_model_config
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    from safetensors import safe_open
    from safetensors.torch import load_file as load_torch_file
    from safetensors.torch import save_file

    destination = Path(destination)
    config = json.loads((destination / "config.json").read_text())
    provenance = config["encoder"]
    if checkpoint_path is None:
        checkpoint_path = hf_hub_download(
            provenance["repo_id"], "model.safetensors", revision=provenance["revision"]
        )
    architecture = load_model_config()
    architecture.text_encoder = None
    architecture.temperature = 1.0
    architecture = OmegaConf.create(OmegaConf.to_container(architecture, resolve=True))
    with init_empty_weights(include_buffers=True):
        model = get_model(config=architecture, pretrained=False, image_only=True)
    expected = model.state_dict()
    tensors = load_torch_file(destination / "model.safetensors")
    with safe_open(checkpoint_path, framework="pt", device="cpu") as weights:
        names = {
            name
            for name in weights.keys()
            if name.startswith("image_encoder.") or name == "softmax_temperature"
        }
        if names != set(expected) or any(
            weights.get_slice(name).get_shape() != list(expected[name].shape) for name in names
        ):
            raise ValueError("Checkpoint does not contain the complete COLIPRI image encoder")
        for name in sorted(names):
            tensors["encoder." + name] = weights.get_tensor(name)
    save_file(tensors, destination / "model.safetensors", metadata={"format": "pt"})
    config["encoder_bundled"] = True
    config["encoder_model_config"] = OmegaConf.to_container(architecture, resolve=True)
    config["dtype"] = "float32"
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    license_path = hf_hub_download(
        provenance["repo_id"], "LICENSE", revision=provenance["revision"]
    )
    shutil.copyfile(license_path, destination / "COLIPRI_LICENSE")


def build_release(destination, *, include_encoder=False, encoder_checkpoint=None, bundled_checkpoint=None):
    """Export only weights, loading metadata, licences and the model card to the Hub."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Use an empty destination for a weight-only Hub export")
    source = Path(bundled_checkpoint) if bundled_checkpoint else ROOT / "artifacts"
    config = json.loads((source / "config.json").read_text())
    if bundled_checkpoint and not config.get("encoder_bundled"):
        raise ValueError("--bundled-checkpoint requires a complete encoder checkpoint directory")
    tensors = load_file(ROOT / "artifacts/model.safetensors")
    config.pop("auto_map", None)
    config.update(
        {
            "model_type": "decision-model",
            "architectures": ["RevModel"],
            "tensor_specs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in tensors.items()
            },
        }
    )
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "processor_config.json").write_text(
        json.dumps({"processor_class": "RevProcessor", "encoder": config["encoder"],
                    "preprocessing": "reference"}, indent=2) + "\n"
    )
    shutil.copyfile(source / "model.safetensors", destination / "model.safetensors")
    for filename in ["LICENSE", "NOTICE", "COLIPRI_LICENSE"]:
        shutil.copyfile(ROOT / filename, destination / filename)
    card = ROOT / "docs/model-card.md"
    (destination / "README.md").write_text((card if card.exists() else ROOT / "README.md").read_text().replace("(../assets/", "(assets/"))
    if (ROOT / "assets").exists():
        shutil.copytree(ROOT / "assets", destination / "assets")
    if include_encoder and not bundled_checkpoint:
        bundle_encoder(destination, encoder_checkpoint)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    parser.add_argument("--bundle-encoder", action="store_true")
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--bundled-checkpoint", type=Path, help="Reuse a complete local checkpoint without changing weights")
    args = parser.parse_args()
    if not args.bundle_encoder and not args.bundled_checkpoint:
        parser.error("A production Hub export requires --bundle-encoder or --bundled-checkpoint")
    if args.encoder_checkpoint and not args.bundle_encoder:
        parser.error("--encoder-checkpoint requires --bundle-encoder")
    if args.bundled_checkpoint and args.encoder_checkpoint:
        parser.error("Use either --bundled-checkpoint or --encoder-checkpoint")
    build_release(args.destination, include_encoder=args.bundle_encoder,
                  encoder_checkpoint=args.encoder_checkpoint, bundled_checkpoint=args.bundled_checkpoint)
