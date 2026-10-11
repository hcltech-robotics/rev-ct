"""Profile upstream CT processing and compare copy-free / direct-ITK candidates."""

import json
import time

import SimpleITK as sitk
import torch
import torchio as tio
from colipri import get_processor
from huggingface_hub import hf_hub_download

NAMES = [
    "valid_675_a_2.nii.gz",
    "valid_740_a_2.nii.gz",
    "valid_458_a_1.nii.gz",
    "valid_901_a_2.nii.gz",
    "valid_851_a_2.nii.gz",
    "valid_74_a_1.nii.gz",
    "valid_114_a_2.nii.gz",
    "valid_361_a_2.nii.gz",
]


def no_copy_processor():
    processor = get_processor(image_only=True)
    processor._image_transform.copy = False
    for transform in processor._image_transform.transforms:
        transform.copy = False
    return processor


def direct_itk(path, processor):
    # Follow the upstream full resampling grid; intensity and crop remain upstream.
    image = sitk.DICOMOrient(sitk.ReadImage(path), "RAS")
    reference = tio.Resample.get_reference_image(image, (2, 2, 2))
    result = sitk.Resample(image, reference, sitk.Transform(), sitk.sitkLinear)
    array, affine = tio.data.io.sitk_to_nib(result)
    scalar = tio.ScalarImage(tensor=torch.from_numpy(array), affine=affine)
    for transform in processor._image_transform.transforms[2:]:
        scalar = transform(scalar)
    return scalar.data


def main():
    torch.set_num_threads(1)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    reference = get_processor(image_only=True)
    fast = no_copy_processor()
    result = {"scope": "same 8 corrected scans; CPU only", "samples": []}
    for name in NAMES:
        parts = name.split("_")
        path = hf_hub_download(
            "ibrahimhamamci/CT-RATE",
            f"dataset/valid_fixed/valid_{parts[1]}/valid_{parts[1]}_{parts[2]}/{name}",
            repo_type="dataset",
            revision="deeca4d89e9f978d4d1bccd88a55071ddbb146bb",
        )
        row = {"volume": name, "modes": {}}
        start = time.perf_counter()
        scalar = tio.ScalarImage(path)
        scalar.load()
        row["load_s"] = time.perf_counter() - start
        row["input_shape"] = list(scalar.shape)
        row["stages"] = []
        for transform in reference._image_transform.transforms:
            start = time.perf_counter()
            scalar = transform(scalar)
            row["stages"].append(
                {"name": type(transform).__name__, "seconds": time.perf_counter() - start}
            )
        for mode in ["reference", "no_copy", "direct_itk_1", "direct_itk_4"]:
            sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(4 if mode.endswith("4") else 1)
            start = time.perf_counter()
            if mode.startswith("direct"):
                tensor = direct_itk(path, fast)
            else:
                processor = reference if mode == "reference" else fast
                tensor = processor.process_images(path)[0].data
            duration = time.perf_counter() - start
            if mode == "reference":
                original = tensor
            row["modes"][mode] = {
                "seconds": duration,
                "max_pixel_error": float(torch.max(torch.abs(tensor - original))),
                "changed_voxels": int(torch.count_nonzero(tensor != original)),
            }
        result["samples"].append(row)
        print("REV_PREPROCESS_RESULT " + json.dumps(result, separators=(",", ":")), flush=True)
    print("REV_PREPROCESS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
