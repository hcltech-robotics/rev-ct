# Benchmarks

Measurements use an NVIDIA A100-SXM4 80 GB on Hugging Face Jobs with 12 allocated
vCPU, PyTorch 2.9.1, CUDA 12.8 and Python 3.11.14. The sample contains 64 distinct
CT-RATE patients: 16 each of 512-matrix, 1024-matrix, small-FOV and thick-slice
acquisitions. Sampling uses seed 0, the corrected `valid_fixed` volumes and the
official no-chest exclusions. All models use the same scans.

## Volume inference

Warm latency starts at reading a local NIfTI and ends at CPU-visible output.
It includes preprocessing, transfer, inference and postprocessing. Downloads,
model loading and initial inference are excluded. Each scan has one timed pass
per protocol, with batch size one, synchronised CUDA, TF32 disabled and one
PyTorch/SimpleITK CPU thread. Four disjoint 16-scan samples form the 64-scan
cohort. The acquisition-balanced sample oversamples uncommon protocols.

| Model / protocol | Output | Precision | Mean s/scan | Median s/scan | p95 s/scan | Serial scans/s |
|---|---|---|---:|---:|---:|---:|
| **RevCT, fast preprocessing** | **18 typed decisions** | **fp32** | **2.856** | **2.667** | **4.201** | **0.350** |
| RevCT, reference preprocessing | 18 typed decisions | fp32 | 7.779 | 6.953 | 13.811 | 0.129 |
| NV-Reason-CT, 18 Yes/No prompts | 18 binary answers | bf16 | 21.796 | 21.292 | 25.744 | 0.0459 |
| COLIPRI zero-shot, paired prompts | 18 similarity scores | bf16 | 6.962 | 6.186 | 13.281 | 0.144 |
| NV-Reason-CT, one list prompt | Free text | bf16 | 7.359 | 6.365 | 11.858 | 0.136 |
| Pillar-0 ChestCT | Vision features | fp32 | 2.521 | 2.381 | 3.626 | 0.397 |
| Pillar-0 ChestCT | Vision features | AMP bf16 | 2.064 | 1.922 | 3.185 | 0.484 |

Fast-preprocessing RevCT is **7.63× faster** than NV's 18-question protocol.
NV's list output and Pillar's features have different output contracts from
RevCT's 18 calibrated finding decisions. The list prompt produced no valid
18-key Boolean JSON mappings in 64 responses. All 1,152 per-label responses
were Yes or No. These speed protocols are separate from the published
classification accuracy results.

NV uses `flash-linear-attention==0.5.2`, `causal-conv1d==1.7.0`, greedy decoding,
thinking disabled and SDPA full attention. The per-label protocol allows eight
tokens per question; the list protocol allows 512. Each volume is preprocessed
once and reused across its questions. Initial binary inference, including
kernel compilation, takes 49.79–52.19 seconds. Warm timings exclude that cost.

COLIPRI caches 36 text embeddings: one negative and one positive prompt for
each finding. It uses the upstream softmax temperature and one image-encoder
pass per scan. Prompts and per-volume timings are in the raw worker reports.

## Preprocessing and stream throughput

The direct SimpleITK processor reduces mean CPU preprocessing from **6.884 to
1.961 seconds (3.51×)** and serial latency by **2.72×**. It avoids redundant
image conversions and copies while preserving the reference resampling grid,
interpolation dtype, HU transforms and crop/pad. All voxels match the reference
on the 64 timed scans. Oblique/sheared geometry and unsupported readers fall
back to the reference processor.

Four bounded preprocessing workers deliver **0.385 scans/s** with the reference
processor and **0.701 scans/s** with the fast processor, including pool startup
for each 16-scan stream. All 1,152 decisions and calibrated scores match the
corresponding serial path. Fast processing with four workers gives **5.45×**
the reference serial throughput. These are single-GPU stream measurements.

```bash
rev-ct predict scan1.nii.gz scan2.nii.gz --preprocessing fast --workers 4 --batch-size 1 --output decisions.jsonl
```

With Transformers, select `AutoProcessor.from_pretrained(repo,
preprocessing="fast")` after importing `rev_ct`. The worker pool uses spawned
processes, one CPU thread per worker and at most twice the worker count in
outstanding preprocessing tasks.

## Encoder precision

Encoder-only measurements use eight real, preprocessed scans resident on the
GPU. They exclude preprocessing, transfer and decision inference. This
experiment uses `dynamic-network-architectures==0.4.4`; the volume comparison
uses 0.4.3.

| Encoder mode | Median seconds | Speedup | Maximum calibrated-score change | Changed decisions / 144 |
|---|---:|---:|---:|---:|
| Eager fp32, TF32 disabled | 0.8862 | 1.00× | reference | 0 |
| TF32 | 0.7024 | 1.26× | 0.01334 | 0 |
| AMP bf16, fp32 weights | 0.1553 | 5.71× | 0.07874 | 0 |
| Compiled fp32 | 0.8660 | 1.02× | 0 | 0 |

The supported model path remains fp32. Approximate precision changes calibrated
scores. Compilation adds 23.67 seconds of first-call startup for approximately
0.020 seconds saved per warm call. The CLI's experimental bf16 option casts
weights; the table measures autocast with fp32 weights.

Reference decision inference for eight embeddings takes a median **0.406 ms**;
the algebraically folded mode takes **0.366 ms**. Folding preserves all cached
validation decisions but changes calibrated scores by up to 0.001288.

## Pillar-0 ChestCT processor

Pillar uses [RAVE](https://github.com/YalaLab/rave)'s `configs/ct_chest.yaml`
and the [chest fine-tuning recipe](https://github.com/YalaLab/pillar-finetune):
256³ at 1.25 mm spacing, symmetric zero-HU depth padding and 11 CT windows.
The window order is lung, mediastinum, abdomen, liver, bone, brain, subdural,
stroke, temporal_bone, soft_tissue and minmax. Input is a local HU NIfTI, without
archive/video conversion or additional mean/std normalisation.

The output contains 1,152 vision features. No CT-RATE finding head is attached
in this benchmark. Across 64 scans, AMP's minimum feature cosine similarity
to fp32 is 0.9999482 and maximum coordinate change is 0.0052082. Pillar runs
in a separate Transformers 4.53.2 environment with fp32 weights and autocast.
Its ECL-2.0 weights are not distributed with RevCT.

## Numerical validation

On 3,039 cached validation embeddings, the safetensors decision model preserves
**54,702/54,702 decisions**, with zero raw-probability error and maximum
calibrated-score error **5.96e-8**. Corrected-cohort macro-F1 is **0.6154176**
and macro-AUROC is **0.8715259**. Full-cohort macro-F1 is **0.6113061** and
macro-AUROC is **0.8655062**. Biases and calibration use training data only.

Across 64 corrected NIfTI scans, fp32 inference preserves all 1,152 cached
decisions, with maximum feature difference 0.00024414. The complete checkpoint
also preserves all 144 scores and decisions in an eight-scan GPU save/reload
check. Native and Transformers inference, the fast processor, strict-schema
records and typed requests have matching outputs in the recorded API checks.

## Reproduction

- [64-scan comparison](benchmark_64.json), with raw `benchmark_64_worker_*.json`
  and `fast_preprocessing_64_worker_*.json` reports.
- [Preprocessing stages](preprocessing_profile.json) and [timed-scan geometry](fast_preprocessing_geometry_64.json).
- [Encoder precision measurements](encoder_optimizations.json).
- [Pillar volume measurements](pillar_volumes_64.json), with raw `pillar_volumes_64_worker_*.json` reports.
- [Validation embedding results](conversion_validation.json).
- [Complete-checkpoint GPU validation](bundled_model_gpu.json) and [offline CPU validation](bundled_model_cpu.json).
- [Transformers outputs](transformers_volume.json) and [fast-processor outputs](transformers_fast_volume.json).
- [Data and precision](provenance.md).

Run `benchmarks/run_benchmark.py` for serial comparisons,
`benchmarks/fast_preprocessing_timing.py` for fast preprocessing,
`benchmarks/rev_optimizations.py` for encoder precision and
`benchmarks/pillar_volumes.py` for Pillar volume inference.
`benchmarks/aggregate_results.py` combines the worker reports.
`python scripts/plot_comparisons.py` regenerates the comparison figures.
