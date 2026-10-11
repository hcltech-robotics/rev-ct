"""Verify the release's public Transformers and typed APIs on one real CT."""

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jsonschema import validate
from safetensors.numpy import load_file
from transformers import AutoModel, AutoProcessor

from rev_ct.export import SOURCE_REPO, export_artifact
from rev_ct.release import systemone


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    revision = "e93a9ecd27e4153bab56258bdd97f7c4668ff9e2"
    source = joblib.load(
        hf_hub_download(SOURCE_REPO, "typed_calibration.joblib", revision=revision)
    )
    destination = Path("/tmp/rev-transformers")
    export_artifact(source, destination, source_revision=revision)
    config = json.loads((destination / "config.json").read_text())
    config.update(
        {
            "model_type": "rev_ct",
            "architectures": ["RevModel"],
            "auto_map": {
                "AutoConfig": "configuration_rev_ct.RevConfig",
                "AutoModel": "modeling_rev_ct.RevModel",
            },
            "tensor_specs": {
                k: {"shape": list(v.shape), "dtype": str(v.dtype)}
                for k, v in load_file(destination / "model.safetensors").items()
            },
        }
    )
    (destination / "config.json").write_text(json.dumps(config))
    (destination / "processor_config.json").write_text(
        json.dumps(
            {
                "processor_class": "RevProcessor",
                "encoder": config["encoder"],
                "auto_map": {"AutoProcessor": "processing_rev_ct.RevProcessor"},
            }
        )
    )
    source_dir = Path(__import__("rev_ct").__file__).parent
    for filename in [
        "configuration_rev_ct.py",
        "modeling_rev_ct.py",
        "processing_rev_ct.py",
        "head.py",
        "inference.py",
        "preprocessing.py",
        "release.py",
    ]:
        shutil.copyfile(source_dir / filename, destination / filename)
    name = "valid_675_a_2.nii.gz"
    path = hf_hub_download(
        "ibrahimhamamci/CT-RATE",
        f"dataset/valid_fixed/valid_675/valid_675_a/{name}",
        repo_type="dataset",
        revision="deeca4d89e9f978d4d1bccd88a55071ddbb146bb",
    )
    model = AutoModel.from_pretrained(destination, trust_remote_code=True).eval().cuda()
    processor = AutoProcessor.from_pretrained(destination, trust_remote_code=True)
    inputs = processor(images3d=[path], return_tensors="pt")
    with torch.inference_mode():
        output = model(**inputs, volume_names=[name])
    fast_processor = AutoProcessor.from_pretrained(
        destination, trust_remote_code=True, preprocessing="fast"
    )
    fast_inputs = fast_processor(images3d=[path], return_tensors="pt")
    torch.testing.assert_close(fast_inputs["pixel_values"], inputs["pixel_values"], rtol=0, atol=0)
    with torch.inference_mode():
        fast_output = model(**fast_inputs, volume_names=[name])
    torch.testing.assert_close(fast_output.probabilities, output.probabilities, rtol=0, atol=0)
    assert torch.equal(fast_output.decisions, output.decisions)
    schema_path = hf_hub_download(SOURCE_REPO, "typed_schema.json", revision=revision)
    validate(output.records[0], json.loads(Path(schema_path).read_text()))
    features = model._inference.encode(inputs["pixel_values"])
    native = model._rev_head().predict(features)
    assert np.array_equal(native.decisions, output.decisions.cpu().numpy())
    assert np.array_equal(native.scores, output.probabilities.cpu().numpy())
    assert model._inference.dtype == torch.float32
    typed = systemone(
        model,
        processor,
        {
            "state": {"volume": path},
            "questions": {"effusion": {"type": "noul", "finding": "Pleural effusion"}},
        },
    )
    result = {
        "volume": name,
        "data_variant": "valid_fixed",
        "gpu": torch.cuda.get_device_name(),
        "encoder_dtype": str(model._inference.dtype),
        "schema_valid": True,
        "native_vs_transformers_score_error": 0,
        "native_vs_transformers_decision_disagreements": 0,
        "fast_processor_changed_voxels": 0,
        "fast_processor_score_error": 0,
        "fast_processor_decision_disagreements": 0,
        "typed_answer": typed["answers"]["effusion"],
    }
    print("REV_TRANSFORMERS_VOLUME " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
