# Changelog

## 0.3.2

- Add the two-author software citation.
- Consolidate inference, model and benchmark documentation.

## 0.3.1

- Add CT-RATE classification, inference-speed and model-size figures.

## 0.3.0

- Separate GitHub package/code releases from Hugging Face weights.
- Register `decision-model` with installed Transformers Auto classes, preserving
  compatibility with earlier `rev_ct` checkpoints.
- Save weights and metadata without copying Python modules.
- Add structured accuracy/timing metadata, NVIDIA's complete comparison table,
  the 64-patient speed comparison and reproducible monochrome figures.
- Preserve encoder weights, decision rules and calibration.

## 0.2.0

- Bundle the frozen COLIPRI image encoder and Rev tensors into one checkpoint.
- Verify offline loading, registered-device inference and complete save/reload.

## 0.1.0

- Export calibrated heads to safetensors with typed decisions and Transformers wrappers.
- Add fast NIfTI preprocessing and matched-volume inference benchmarks.
