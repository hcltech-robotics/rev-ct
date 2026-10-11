import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
import torch
from colipri import get_processor
from transformers import AutoProcessor

from rev_ct.preprocessing import FastImageProcessor


@pytest.mark.parametrize("dtype", [np.int16, np.float32])
@pytest.mark.parametrize("orientation", ["flipped", "permuted", "oblique", "sheared"])
def test_fast_preprocessing_preserves_reference_pixels(tmp_path, dtype, orientation):
    torch.set_num_threads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    affine = np.diag([-3.0, 2.3, -4.0, 1.0])
    if orientation == "permuted":
        affine[:3, :3] = affine[[2, 0, 1], :3]
    elif orientation == "oblique":
        theta = 0.3
        rotation = np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        )
        affine[:3, :3] = rotation @ affine[:3, :3]
    elif orientation == "sheared":
        affine[0, 1] = 0.2
    affine[:3, 3] = [40, -30, 50]
    data = (
        np.random.default_rng(0)
        .integers(-1200, 1800, size=(1, 24, 28, 32), dtype=np.int16)
        .astype(dtype)
    )
    if dtype == np.float32:
        data += np.float32(0.25)
    path = tmp_path / "ct.nii.gz"
    nib.save(nib.Nifti1Image(data[0], affine), path)
    original = get_processor(image_only=True).process_images(str(path))[0].data
    fast = FastImageProcessor().process_images(str(path))[0].data
    torch.testing.assert_close(fast, original, rtol=0, atol=0)


def test_fast_auto_processor_option_survives_save_reload(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    processor = AutoProcessor.from_pretrained(
        root / "artifacts", trust_remote_code=True, preprocessing="fast"
    )
    assert processor.preprocessing == "fast"
    processor.save_pretrained(tmp_path)
    reloaded = AutoProcessor.from_pretrained(tmp_path, trust_remote_code=True)
    assert reloaded.preprocessing == "fast"
