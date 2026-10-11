# Inference

Rev uses a fixed vocabulary of 18 CT-RATE findings. Each has a trained
present/absent rule and a calibrated score. The decision uses an F1-selected
training bias. The score estimates the label probability after isotonic
calibration. Do not reconstruct decisions by thresholding calibrated scores.

## Load the complete model

Install `rev-ct[inference]` from the GitHub release. Importing `rev_ct`
registers the installed Transformers architecture.

```python
import rev_ct
import torch
from transformers import AutoModel, AutoProcessor

repo = "hcltech-robotics/rev-ct"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
model = AutoModel.from_pretrained(repo).eval().to("cuda")
processor = AutoProcessor.from_pretrained(repo, preprocessing="fast")
inputs = processor(images3d=["chest.nii.gz"], return_tensors="pt")
with torch.inference_mode():
    output = model(**inputs, volume_names=["chest.nii.gz"])
```

Use the same pinned Hub revision for both objects in production. The image
path is fp32. Disable CUDA matmul and cuDNN TF32 for parity with the measured
reference. Global loading dtypes, `.half()` and `.bfloat16()` change the mixed
head dtypes and are unsupported. Experimental encoder-only precision
settings have separate score-drift measurements in the benchmark report.

The bundled checkpoint contains the COLIPRI image encoder and omits the text
encoder. It requires the installed COLIPRI architecture and processor, but
makes no second encoder-weight download. There is no remote Python loading.

## Inputs and preprocessing

Volumes need HU intensities and correct physical geometry. DICOM series need
prior conversion to NIfTI. The processor reorients to RAS, resamples to 2 mm,
clamps HU to [-1000, 1000], rescales to [-1, 1], and centres a 192³ crop/pad.
It does not implement an anatomical lung crop.

`preprocessing="reference"` is the default TorchIO path. The explicit `fast`
processor uses direct SimpleITK loading/resampling, preserving source pixel
type and reference transforms. Oblique/sheared affine geometry and unsupported
readers fall back to the reference. All voxels match on 64 tested scans.
For CT-RATE, use corrected `dataset/{train,valid}_fixed/` files.

## Output contract

| Field | Shape / type | Meaning |
|---|---|---|
| `logits` | `(N, 18)` tensor | Original logits minus training-selected decision biases |
| `raw_probabilities` | `(N, 18)` tensor | Original sigmoid probabilities, used for AUROC |
| `probabilities` | `(N, 18)` tensor | Isotonic-calibrated scores |
| `decisions` | `(N, 18)` Boolean tensor | `logits >= 0` |
| `records` | List of dictionaries | JSON-compatible `rev-1.0` typed records |

The schema is in [typed_schema.json](../artifacts/typed_schema.json). Each
finding has a `decision`, `score` and `confidence`. Confidence is the score
when present and one minus the score when absent. It is not separately
calibrated against decision correctness under the F1-selected threshold.
The record also has a top-finding summary, mean-score `burden` and `n_positive`.
Burden is not a clinically validated severity measure.

## Typed requests

The adapter follows [Clef](https://huggingface.co/cloudflare/clef)'s loading
and typed-answer conventions. It accepts `noul` questions about trained
findings, rather than arbitrary natural-language questions.

```python
from rev_ct.release import load_release_model, systemone

model, processor = load_release_model(device="cuda")
response = systemone(model, processor, {
    "model": "rev-ct",
    "state": {"volume": "chest.nii.gz"},
    "questions": {
        "effusion": {"type": "noul", "finding": "Pleural effusion"},
    },
})
print(response["answers"]["effusion"])
```

Omit `questions` for all 18 findings. Unsupported question types and finding
names are rejected before encoding. `encode_record` and `collate_records`
support batches of encoded records, with one volume per typed request.

## Cached features and CLI

```python
output = model(features=torch.tensor(features, dtype=torch.float32))
```

Features must be pooled, projected, unnormalised 768-dimensional embeddings
from the pinned COLIPRI encoder. Image inference rounds embeddings through
fp16 and back to fp32, matching training storage. Normalised contrastive
embeddings and features from other encoders are incompatible.

```bash
rev-ct predict chest.nii.gz --output decisions.jsonl
rev-ct predict scan1.nii.gz scan2.nii.gz --preprocessing fast --workers 4 --batch-size 1 --output decisions.jsonl
rev-ct features cached.npz --output decisions.jsonl
```

NPZ inputs accept `features` or `feats`, plus `names`. Four preprocessing
workers feed one GPU, reaching 0.701 scans/s in the recorded timing sample.
Choose worker count to fit available CPU and RAM. The core `RevHead` class
supports feature-only inference without imaging dependencies.

## Save and use offline

```python
model.save_pretrained("rev-local")
processor.save_pretrained("rev-local")
model = AutoModel.from_pretrained("rev-local", local_files_only=True).eval().to("cuda")
processor = AutoProcessor.from_pretrained("rev-local", local_files_only=True)
```

Saved directories contain the complete safetensors checkpoint and loading
metadata, without copied Python modules. Install the same `rev_ct` package
version and import it before loading.

## Numerical evidence

On 3,039 cached validation volumes, the safetensors heads preserve all 54,702
decisions, with maximum score error 5.96e-8. On eight real scans the complete
bundled model exactly matches the original Rev image path, including all
144 scores/decisions and GPU save/reload. Across 64 NIfTI inputs all 1,152
decisions match cached decisions. Small feature drift can change calibrated
scores on steep isotonic segments. Full-cohort NIfTI parity is not established.

Optional `head_mode="fused"` preserves all cached validation decisions but
has maximum calibrated-score drift 0.00129 and negligible end-to-end speed
benefit. Reference mode remains the default.
