---
title: RevCT
emoji: ◼️
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: '3.12'
app_file: app.py
fullWidth: true
license: apache-2.0
models:
- hcltech-robotics/rev-ct
tags:
- decision-model
- chest-ct
- medical-imaging
- multi-label-classification
short_description: Calibrated typed decisions for chest CT
---

# RevCT

A typed decision model for computed tomography (CT).

Explore a chest CT in axial, coronal and sagittal views, obtain the 18 CT-RATE finding decisions and download the complete calibrated decision record.

[Model and weights](https://huggingface.co/hcltech-robotics/rev-ct) · [Code](https://github.com/hcltech-robotics/rev-ct) · [Local demo](https://github.com/hcltech-robotics/rev-ct/tree/main/demo) · [Citation](https://doi.org/10.57967/hf/10861)

The public example's decisions were computed with the complete RevCT v0.3.2 checkpoint and are precomputed for immediate display. Uploaded NIfTI scans run live with ZeroGPU. Preprocessing runs on CPU; the model uses fp32 inference and reserves GPU time only for its forward pass.

RevCT is a research model and has not been clinically validated. Do not use it for diagnosis or treatment. Upload only de-identified scans that you have permission to share. Run the demo locally for private data.

The PyRadiomics `lung1` example is subject to the source collection's separate [CC BY-NC 3.0 terms](https://creativecommons.org/licenses/by-nc/3.0/); attribution and conversion details are in [demo/examples/NOTICE.md](demo/examples/NOTICE.md).

Rev was developed at the HCLTech Robotics Intelligence CoE.

## Citation

```bibtex
@software{voncsefalvay2026revct,
  author  = {von Csefalvay, Chris and Foldi, Tamas},
  title   = {RevCT: A typed decision model for computed tomography (CT)},
  year    = {2026},
  version = {0.3.2},
  doi     = {10.57967/hf/10861},
  url     = {https://github.com/hcltech-robotics/rev-ct}
}
```
