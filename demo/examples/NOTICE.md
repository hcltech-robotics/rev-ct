# Example CT

`lung1.nii.gz` is the PyRadiomics `lung1` example image, obtained from the [PyRadiomics v1.0 release](https://github.com/Radiomics/pyradiomics/releases/download/v1.0/lung1_image.nrrd). It was converted from NRRD to NIfTI without changing its CT intensities or physical geometry; auxiliary header metadata was removed.

The source collection is [NSCLC-Radiomics (Lung1), The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/nsclc-radiomics/), DOI [10.7937/K9/TCIA.2015.PF0M9REI](https://doi.org/10.7937/K9/TCIA.2015.PF0M9REI). The collection's current data terms are [Creative Commons Attribution–NonCommercial 3.0](https://creativecommons.org/licenses/by-nc/3.0/). These terms apply to this example image and its derived decision record, separately from the RevCT code licence.

The cached decision record is an actual fp32 inference result from RevCT v0.3.2. It contains model predictions, rather than clinical reference labels.
