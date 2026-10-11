"""Pillar ChestCT feature extraction using the published RVE chest recipe."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import rve
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml
from run_benchmark import DATA_REVISION, load_data
from transformers import AutoModel
from vision_engine.processing.ct_processor import CTProcessor

from rev_ct.benchmark import summarise

REPO = "YalaLab/Pillar0-ChestCT"
REVISION = "b9dfd833a947744382d4f1bd73b5dd72fcb58b34"
RVE_REVISION = "20adeb873021c864e9410fae8898b0c8da309769"
FINETUNE_REVISION = "47506d5267cb448867fe415d416910ce6aad9422"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cohort, _, _ = load_data(Path("/tmp/pillar-data"), 64, 0, args.worker_index, 4)
    config_data = yaml.safe_load(Path("/tmp/rave/configs/ct_chest.yaml").read_text())
    # The wheel omits configs/exporters, although Config resolves them relative
    # to its installed module. Resolve the published exporter file in the clone.
    config_data["exporter_config"] = "/tmp/rave/configs/exporters/video_hevc.yaml"
    config = rve.Config(**config_data)
    processor = CTProcessor(config)
    windows = rve.get_available_windows("CT")
    assert len(windows) == 11
    started = time.perf_counter()
    model = AutoModel.from_pretrained(REPO, revision=REVISION, trust_remote_code=True).eval().cuda()
    torch.cuda.synchronize()
    report = {
        "repo": REPO,
        "revision": REVISION,
        "rve_repo": "https://github.com/YalaLab/rave",
        "rve_revision": RVE_REVISION,
        "finetune_revision": FINETUNE_REVISION,
        "data_revision": DATA_REVISION,
        "worker_index": args.worker_index,
        "gpu": torch.cuda.get_device_name(),
        "flavour": "a100-large",
        "allocated_vcpu": 12,
        "packages": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "timm", "SimpleITK", "torchio"]},
        "scope": "local corrected NIfTI through CPU-visible 1152-d vision features; no finding head or lossy archive export",
        "preprocessing": {"config": "configs/ct_chest.yaml", "processing": config.processing, "depth_padding": "symmetric zero HU before windowing, as pillar-finetune CSV dataset", "windows": windows, "extra_normalization": False},
        "load_s": time.perf_counter() - started,
        "results": {},
        "cohort": [{k: v for k, v in r.items() if k != "path"} for r in cohort],
    }

    def forward(path, amp):
        torch.cuda.synchronize()
        started = time.perf_counter()
        image = sitk.ReadImage(str(path))
        volume = processor._process_volume(image, np.array(image.GetSpacing()))
        pixels = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        if pixels.shape[2] < 256:
            extra = 256 - pixels.shape[2]
            pixels = F.pad(pixels, (0, 0, 0, 0, extra // 2, extra - extra // 2))
        assert tuple(pixels.shape) == (1, 1, 256, 256, 256)
        prep = time.perf_counter() - started
        pixels = pixels.cuda()
        torch.cuda.synchronize()
        transferred = time.perf_counter()
        # This follows pillar-finetune's CT batch_windowing; CT has no
        # additional mean/std normalization. Keep the prescribed window order.
        windowed = torch.stack([rve.apply_windowing(pixels[:, 0], w, "CT") for w in windows], dim=1)
        torch.cuda.synchronize()
        window_end = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            output = model.extract_vision_feats(image={"chest_ct": windowed})
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        features = output.float().cpu().numpy()
        total = time.perf_counter() - started
        assert features.shape == (1, 1152) and np.isfinite(features).all()
        return {"preprocess_s": prep, "h2d_s": transferred - started - prep, "windowing_s": window_end - transferred, "encoder_s": encoded - window_end, "total_s": total}, features

    reference = {}
    for mode, amp in [("fp32", False), ("amp_bf16", True)]:
        first, _ = forward(cohort[0]["path"], amp)
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for row in cohort:
            sample, features = forward(row["path"], amp)
            sample["volume"] = row["name"]
            if not amp:
                reference[row["name"]] = features
            else:
                baseline = reference[row["name"]]
                sample["max_feature_error"] = float(np.max(abs(features - baseline)))
                sample["feature_cosine_similarity"] = float(np.dot(features.ravel(), baseline.ravel()) / (np.linalg.norm(features) * np.linalg.norm(baseline)))
            samples.append(sample)
            print("PILLAR_VOLUME_SAMPLE " + json.dumps({"mode": mode, **sample}), flush=True)
        report["results"][mode] = {"first_inference_s": first["total_s"], "summary": summarise(samples), "samples": samples, "peak_vram_gib": torch.cuda.max_memory_allocated() / 1024**3}
        print("PILLAR_VOLUMES_RESULT " + json.dumps(report, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
