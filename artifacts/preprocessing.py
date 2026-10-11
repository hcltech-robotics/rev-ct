"""Exact upstream volume geometry with fewer image conversions and copies."""

from __future__ import annotations


class FastImageProcessor:
    """Process fresh NIfTI paths; preserve the source dtype during interpolation."""

    def __init__(self):
        from colipri import get_processor

        self.reference = get_processor(image_only=True)
        for transform in self.reference._image_transform.transforms:
            transform.copy = False

    def process_images(self, paths):
        import numpy as np
        import SimpleITK as sitk
        import torch
        import torchio as tio

        if not isinstance(paths, list):
            paths = [paths]
        images = []
        for path in paths:
            try:
                image = sitk.ReadImage(str(path))
            except RuntimeError:
                # TorchIO can fall back to NiBabel for NIfTI affines that ITK
                # cannot read (for example, sheared direction cosines).
                from colipri import get_processor

                images.extend(get_processor(image_only=True).process_images(str(path)))
                continue
            direction = np.asarray(image.GetDirection()).reshape(image.GetDimension(), -1)
            if image.GetDimension() != 3 or not np.all(
                np.isclose(abs(direction), 0, atol=1e-12, rtol=0)
                | np.isclose(abs(direction), 1, atol=1e-12, rtol=0)
            ):
                # Oblique affines can round differently through TorchIO's
                # reorientation/conversion, changing resampling grid dimensions.
                from colipri import get_processor

                images.extend(get_processor(image_only=True).process_images(str(path)))
                continue
            image = sitk.DICOMOrient(image, "RAS")
            grid = tio.Resample.get_reference_image(image, (2, 2, 2))
            resampled = sitk.Resample(image, grid, sitk.Transform(), sitk.sitkLinear)
            array, affine = tio.data.io.sitk_to_nib(resampled)
            scalar = tio.ScalarImage(tensor=torch.from_numpy(array), affine=affine)
            for transform in self.reference._image_transform.transforms[2:]:
                scalar = transform(scalar)
            images.append(scalar)
        return images

    def to_images_batch(self, images):
        return self.reference.to_images_batch(images)
