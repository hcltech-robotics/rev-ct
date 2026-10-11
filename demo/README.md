# RevCT demo

Upload a chest CT in HU-valued NIfTI format (`.nii` or `.nii.gz`), browse its slices and run RevCT to obtain 18 typed finding decisions. The viewer supports axial, coronal and sagittal planes with lung, mediastinal and bone windows. Download the complete `rev-1.0` decision record as JSON.

```bash
python -m pip install -r demo/requirements.txt
python app.py
```

Open http://localhost:7860. The demo loads the complete `hcltech-robotics/rev-ct` checkpoint at `v0.3.2` through Transformers. Inference uses fp32 and the verified fast preprocessing path. The model loads once; requests run one at a time. Local inference uses CUDA when available, with a CPU fallback.

The [hosted demo](https://huggingface.co/spaces/chrisvoncsefalvay/rev-ct-demo) uses ZeroGPU. Its model is placed on CUDA during startup, as required by Spaces. Preprocessing runs on CPU, and a GPU-decorated forward pass reserves up to 30 seconds. Example viewing and cached example decisions do not reserve a GPU.

The supplied example is the openly distributed PyRadiomics `lung1` CT, converted to NIfTI with intensities and physical geometry preserved. Its decisions were computed using the complete RevCT checkpoint and cached for immediate display. Files matching the example's bytes use these precomputed results, including renamed uploads; other scans run live. Example-data terms and attribution are in [examples/NOTICE.md](examples/NOTICE.md).

Only upload de-identified research scans. Temporary uploads and downloads expire from Gradio's cache after 30 minutes. The public bundled example remains available for the lifetime of the app. The app does not log scan contents or use uploads for training.

RevCT has not been clinically validated and must not be relied on for diagnosis or treatment. See the [model card](https://huggingface.co/hcltech-robotics/rev-ct) for the fixed finding vocabulary and calibration limits.
