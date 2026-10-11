"""Measure fast preprocessing on a shared slice of the 64-scan CT cohort."""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import SimpleITK as sitk
import torch
from huggingface_hub import hf_hub_download
from run_benchmark import SOURCE_REVISION, load_data

from rev_ct.benchmark import summarise
from rev_ct.export import SOURCE_REPO, export_artifact
from rev_ct.head import RevHead
from rev_ct.inference import RevInference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = joblib.load(
        hf_hub_download(SOURCE_REPO, "typed_calibration.joblib", revision=SOURCE_REVISION)
    )
    export_artifact(source, "/tmp/rev-head", source_revision=SOURCE_REVISION)
    head = RevHead.from_pretrained("/tmp/rev-head")
    cohort, _, _ = load_data(Path("/tmp/rev-fast-data"), 64, 0, args.worker_index, 4)
    model = RevInference(head, preprocessing="fast")
    from colipri import get_processor

    original = get_processor(image_only=True)
    model.encode(model.preprocess([cohort[0]["path"]]))
    samples, records = [], []
    report = {
        "worker_index": args.worker_index,
        "gpu": torch.cuda.get_device_name(),
        "scope": "same 64-scan cohort, one timed fast forward per volume; reference pixels checked separately",
        "samples": samples,
    }
    for row in cohort:
        reference_pixels = original.to_images_batch(original.process_images(row["path"]))
        torch.cuda.synchronize()
        start = time.perf_counter()
        pixels = model.preprocess([row["path"]])
        prep = time.perf_counter() - start
        torch.cuda.synchronize()
        features = model.encode(pixels)
        torch.cuda.synchronize()
        output = head.records(features, [row["name"]])[0]
        total = time.perf_counter() - start
        record = {
            "volume": row["name"],
            "total_s": total,
            "preprocess_s": prep,
            "changed_voxels": int(torch.count_nonzero(pixels != reference_pixels)),
            "max_pixel_error": float(torch.max(abs(pixels - reference_pixels))),
        }
        records.append(output)
        samples.append(record)
        print("REV_FAST_SAMPLE " + json.dumps(record), flush=True)
    report["summary"] = summarise(samples)
    start = time.perf_counter()
    parallel = list(model.predict([r["path"] for r in cohort], workers=4, batch_size=1))
    wall = time.perf_counter() - start

    def scores(rs):
        return np.array([[d["score"] for d in r["decisions"]] for r in rs])

    report["parallel"] = {
        "wall_s": wall,
        "workers": 4,
        "includes_startup": True,
        "volumes_per_second": len(cohort) / wall,
        "max_score_error": float(np.max(abs(scores(records) - scores(parallel)))),
        "decision_disagreements": sum(
            a["decision"] != b["decision"]
            for ra, rb in zip(records, parallel, strict=True)
            for a, b in zip(ra["decisions"], rb["decisions"], strict=True)
        ),
    }
    print("REV_FAST_RESULT " + json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
