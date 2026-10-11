"""Measure encoder precision/compile candidates and output drift on eight real scans."""

import importlib.metadata
import json
import time

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from rev_ct.export import SOURCE_REPO, export_artifact
from rev_ct.head import RevHead
from rev_ct.inference import RevInference

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


def main():
    import joblib

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source_revision = "e93a9ecd27e4153bab56258bdd97f7c4668ff9e2"
    artifact = joblib.load(
        hf_hub_download(SOURCE_REPO, "typed_calibration.joblib", revision=source_revision)
    )
    export_artifact(artifact, "/tmp/rev-head", source_revision=source_revision)
    head = RevHead.from_pretrained("/tmp/rev-head")
    model = RevInference(head)
    pixels = []
    for name in NAMES:
        parts = name.split("_")
        path = hf_hub_download(
            "ibrahimhamamci/CT-RATE",
            f"dataset/valid_fixed/valid_{parts[1]}/valid_{parts[1]}_{parts[2]}/{name}",
            repo_type="dataset",
            revision="deeca4d89e9f978d4d1bccd88a55071ddbb146bb",
        )
        pixels.append(model.preprocess([path]).cuda())
    result = {
        "gpu": torch.cuda.get_device_name(),
        "names": NAMES,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["torch", "colipri", "dynamic-network-architectures"]
        },
        "scope": "encoder only; preprocessed inputs resident on GPU; 8 scans x 2 timed repeats",
        "results": {},
    }

    def run(encoder, *, autocast=False):
        features, times = [], []
        start = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            encoder(pixels[0], pool=True, project=True, normalize=False)
        torch.cuda.synchronize()
        first = time.perf_counter() - start
        for _ in range(2):
            for image in pixels:
                torch.cuda.synchronize()
                start = time.perf_counter()
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast),
                ):
                    output = encoder(image, pool=True, project=True, normalize=False)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)
                features.append(output.cpu().to(torch.float16).numpy().astype(np.float32))
        return {
            "first_forward_s": first,
            "p50_encoder_s": float(np.median(times)),
            "p95_encoder_s": float(np.percentile(times, 95)),
            "samples_s": times,
        }, np.concatenate(features)

    eager, reference = run(model.model.image_encoder)
    result["results"]["fp32"] = eager
    original = head.predict(reference)
    for name in ["tf32", "amp_bf16", "compile_fp32"]:
        try:
            torch.backends.cuda.matmul.allow_tf32 = name == "tf32"
            torch.backends.cudnn.allow_tf32 = name == "tf32"
            encoder = model.model.image_encoder
            if name == "compile_fp32":
                encoder = torch.compile(encoder, mode="reduce-overhead")
            report, features = run(encoder, autocast=name == "amp_bf16")
            candidate = head.predict(features)
            report.update(
                {
                    "decision_disagreements": int(
                        np.sum(original.decisions != candidate.decisions)
                    ),
                    "max_feature_error": float(np.max(abs(reference - features))),
                    "max_score_error": float(np.max(abs(original.scores - candidate.scores))),
                }
            )
            result["results"][name] = report
        except Exception as error:
            result["results"][name] = {"error": type(error).__name__ + ": " + str(error)[:1000]}
        print("REV_OPTIMIZATION_RESULT " + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
