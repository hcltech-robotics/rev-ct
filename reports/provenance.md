# Data and precision

CT-RATE inputs use the corrected `dataset/{train,valid}_fixed/` NIfTI volumes.
The COLIPRI image encoder runs in fp32, with CUDA matmul and cuDNN TF32 disabled.
Pooled, projected features are rounded through fp16 and returned to fp32,
matching the training feature format. Scaling, decision biases and isotonic
calibration preserve their stored tensor types.

Validation uses official RadBERT labels. The corrected cohort contains 3,002
volumes after the 37 official no-chest exclusions; the full cohort contains
3,039 volumes. Decision biases and calibration are fitted to training
out-of-fold probabilities.

See the [model card](../docs/model-card.md) for pinned encoder and dataset
revisions, and the [benchmark report](benchmark.md) for timing protocols.
