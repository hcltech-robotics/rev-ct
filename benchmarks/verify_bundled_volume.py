"""Check bundled Transformers volume inference against the original Rev path."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from jsonschema import validate
from transformers import AutoModel, AutoProcessor

from rev_ct.head import RevHead
from rev_ct.inference import RevInference
from rev_ct.release import systemone

NAMES = [
    "valid_675_a_2.nii.gz",
    "valid_740_a_2.nii.gz",
    "valid_458_a_1.nii.gz",
    "valid_901_a_2.nii.gz",
    "valid_851_a_2.nii.gz",
    "valid_74_a_1.nii.gz",
    "valid_114_a_2.nii.gz",
    "valid_361_a_2.nii.gz",
]
DATA_REVISION = "deeca4d89e9f978d4d1bccd88a55071ddbb146bb"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="chrisvoncsefalvay/rev-ct")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--reference-revision", default="f71d8484b698a1f9a8db3ee460d20f0f33dc013e")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model = AutoModel.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True
    ).eval().cuda()
    assert model.config.encoder_bundled
    assert model.encoder.text_encoder is None
    assert next(model.encoder.parameters()).dtype == torch.float32
    processor = AutoProcessor.from_pretrained(
        args.model, revision=args.revision, trust_remote_code=True, preprocessing="fast"
    )
    reference = RevInference(
        RevHead.from_pretrained(args.model, revision=args.reference_revision), device="cuda"
    )
    reference_buffers = dict(reference.model.named_buffers())
    for name, buffer in model.encoder.named_buffers():
        assert torch.equal(buffer, reference_buffers[name]), name
    schema = json.loads(
        Path(hf_hub_download(args.model, "typed_schema.json", revision=args.revision)).read_text()
    )
    samples = []
    for name in NAMES:
        study, series = name.removesuffix(".nii.gz").rsplit("_", 1)[0].rsplit("_", 1)
        path = hf_hub_download(
            "ibrahimhamamci/CT-RATE",
            f"dataset/valid_fixed/{study}/{study}_{series}/{name}",
            repo_type="dataset",
            revision=DATA_REVISION,
        )
        inputs = processor(images3d=[path], return_tensors="pt")
        features = reference.encode(inputs["pixel_values"])
        expected = reference.head.predict(features)
        with torch.inference_mode():
            output = model(**inputs, volume_names=[name])
        actual = output.probabilities.cpu().numpy()
        assert np.array_equal(actual, expected.scores), name
        assert np.array_equal(output.decisions.cpu().numpy(), expected.decisions), name
        validate(output.records[0], schema)
        samples.append({"name": name, "score_max_error": 0, "decision_disagreements": 0})
        print("BUNDLE_VOLUME " + json.dumps(samples[-1]), flush=True)
    typed = systemone(
        model,
        processor,
        {"state": {"volume": path}, "questions": {"effusion": {"type": "noul", "finding": "Pleural effusion"}}},
    )
    assert typed["answers"]["effusion"]["noul"] == round(float(actual[0, 12]), 4)
    directory = Path("/tmp/rev-bundled-roundtrip")
    model.save_pretrained(directory)
    processor.save_pretrained(directory)
    del model
    reloaded = AutoModel.from_pretrained(
        directory, trust_remote_code=True, local_files_only=True
    ).eval().cuda()
    assert next(reloaded.encoder.parameters()).dtype == torch.float32
    restored = AutoProcessor.from_pretrained(directory, trust_remote_code=True, local_files_only=True)
    assert restored.preprocessing == "fast"
    with torch.inference_mode():
        reloaded_output = reloaded(**inputs, volume_names=[name])
    assert torch.equal(reloaded_output.probabilities, output.probabilities)
    assert torch.equal(reloaded_output.decisions, output.decisions)
    # Also exercise the CLI's native path with the bundled checkpoint.
    native = RevInference(
        RevHead.from_pretrained(directory), device="cuda", preprocessing="fast"
    )
    assert np.array_equal(native.encode(inputs["pixel_values"]), features)
    result = {
        "revision": args.revision,
        "reference_revision": args.reference_revision,
        "gpu": torch.cuda.get_device_name(),
        "encoder_dtype": str(next(reloaded.encoder.parameters()).dtype),
        "encoder_bundled": True,
        "encoder_buffers_equal_reference": True,
        "volumes": samples,
        "n_volumes": len(samples),
        "n_decisions": len(samples) * 18,
        "score_max_error": 0,
        "decision_disagreements": 0,
        "schema_valid": True,
        "typed_api_valid": True,
        "save_pretrained_volume_roundtrip_score_error": 0,
        "save_pretrained_volume_roundtrip_decision_disagreements": 0,
        "native_bundled_encoder_feature_error": 0,
    }
    print("REV_BUNDLED_VOLUME " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
