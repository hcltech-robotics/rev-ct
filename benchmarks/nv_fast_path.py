"""Run NV's 18-label protocols with verified optional linear-attention kernels."""

import importlib.metadata
import json
from pathlib import Path

import joblib
import torch
from huggingface_hub import hf_hub_download
from run_benchmark import SOURCE_REVISION, load_data, nv_benchmark
from transformers.models.qwen3_5 import modeling_qwen3_5

from rev_ct.export import SOURCE_REPO, export_artifact
from rev_ct.head import RevHead


def main():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not modeling_qwen3_5.is_fast_path_available:
        raise RuntimeError("Qwen3.5 linear-attention fast path is unavailable")
    source = joblib.load(
        hf_hub_download(SOURCE_REPO, "typed_calibration.joblib", revision=SOURCE_REVISION)
    )
    export_artifact(source, "/tmp/rev-head", source_revision=SOURCE_REVISION)
    head = RevHead.from_pretrained("/tmp/rev-head")
    cohort, _, _ = load_data(Path("/tmp/nv-fast-data"), 8, 0)
    print("NV_FAST_READY", flush=True)
    result = {
        "gpu": torch.cuda.get_device_name(),
        "linear_attention_fast_path": True,
        "packages": {
            p: importlib.metadata.version(p)
            for p in [
                "torch",
                "transformers",
                "flash-linear-attention",
                "causal-conv1d",
                "dynamic-network-architectures",
            ]
        },
        "settings": {
            "count": 8,
            "repeats": 2,
            "tf32": False,
            "torch_cpu_threads": 1,
            "data_variant": "valid_fixed",
        },
        "cohort": [{k: v for k, v in row.items() if k != "path"} for row in cohort],
        "result": nv_benchmark(head, cohort, 2),
    }
    print("NV_FAST_RESULT " + json.dumps(result, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
