"""Core preprocessing primitives invoked by the batch driver (`main.py`).

Functions:
    convert_dicom_to_nii: Convert a DICOM series directory to a single NIfTI file.
    strip_skull: Run SynthStrip to produce a brain-only image (and mask) from input NIfTI.
    correct_n4_bias: Apply N4 bias field correction (reduces smooth intensity shading).
    normalize: Intensity normalization (z-score / robust / min-max / percentile schemes).
    resample: Resample an image to specified voxel spacing (e.g. isotropic 1 mm³).

Design:
    Each function returns the output file path, enabling straightforward chaining in a pipeline.
    Functionality remains minimal and explicit, delegating heavy lifting to established libraries
    (dicom2nifti, SynthStrip, SimpleITK, nibabel).
"""

import dicom2nifti
import os
import sys
import nibabel as nib
import SimpleITK as sitk
import numpy as np
import torch

from nipreps.synthstrip.cli import main as synthstrip_main
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[2]
DATA_DIR = os.path.join(BASE_PATH, "data")
MODEL_PATH = os.path.join(DATA_DIR, "synthstrip/pytorch/main/1/synthstrip.1.pt")


# Dicom to nii
def convert_dicom_to_nii(dicom_path: str, output_file: str) -> str:
    """Convert a DICOM series directory into a single NIfTI image.

    Args:
        dicom_path: Directory containing DICOM slice files.
        output_file: Destination path for generated NIfTI file (.nii / .nii.gz).

    Returns:
        str: Path to written NIfTI file.
    """
    nifti_path = dicom2nifti.dicom_series_to_nifti(dicom_path, output_file)["NII_FILE"]
    return nifti_path

# Skull Stripping
def strip_skull(nifti_path: str, out_prefix: str) -> str:
    """Run SynthStrip skull stripping producing brain-only image (and mask file).

    Args:
        nifti_path: Input anatomical NIfTI file path.
        out_prefix: Output prefix used for image and mask naming.

    Returns:
        str: Path to skull-stripped brain image.
    """

    out_img = f"{out_prefix}_output.nii.gz"
    out_msk = f"{out_prefix}_mask.nii.gz"

    sys.argv = [
        "synthstrip",
        "--image", nifti_path,
        "--out", out_img,
        "--mask", out_msk,
        "--model", MODEL_PATH
    ]

    if torch.cuda.is_available():
        sys.argv.append("-g")

    synthstrip_main()
    return out_img

# Correction N4 bias
def correct_n4_bias(
    nifti_path: str,
    out_path: str | None = None,
    mask_path: str | None = None,
    shrink_factor: int = 4,
    max_iters: tuple[int, int, int] = (50, 30, 20),
    bspline_fitting_levels: int | None = None,
) -> str:
    """Apply N4 bias field correction to mitigate smooth intensity shading.

    Args:
        nifti_path: Path to input NIfTI image.
        out_path: Optional explicit output path; auto-generated if None.
        mask_path: Optional brain mask path; if None, an Otsu mask is computed.
        shrink_factor: Spatial shrink factor for initial bias estimation (>1 speeds estimation).
        max_iters: Iterations per resolution level for the N4 algorithm.
        bspline_fitting_levels: Override number of B-spline fitting levels if provided.

    Returns:
        str: Path to bias-corrected NIfTI image.
    Raises:
        FileNotFoundError: If input image or provided mask does not exist.
    """
    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(nifti_path)
    if out_path is None:
        base, ext = os.path.splitext(nifti_path)
        # handle .nii.gz double extension
        if base.endswith(".nii"):
            base = base[:-4]
        out_path = base + "_n4.nii.gz"
    # Read image via SimpleITK (preserves header & spacing)
    img = sitk.ReadImage(nifti_path)
    # N4 requires a real (floating) pixel type; cast if necessary (e.g. int16 input)
    if img.GetPixelID() not in (sitk.sitkFloat32, sitk.sitkFloat64):
        img = sitk.Cast(img, sitk.sitkFloat32)
    if mask_path:
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(mask_path)
        mask_img = sitk.ReadImage(mask_path)
        if mask_img.GetPixelID() != sitk.sitkUInt8:
            mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
    else:
        # Otsu on original for rough brain mask
        mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
        mask_img = sitk.Cast(mask_img, sitk.sitkUInt8)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    # SimpleITK N4 filter does NOT expose SetShrinkFactor in many versions; emulate by shrinking input.
    # If shrink_factor > 1 we create downsampled images for estimating the bias then apply to full.
    # Strategy: run on (optionally) shrunken image, then if shrink was applied, re-run on full-res using computed field.
    corrector.SetMaximumNumberOfIterations(max_iters)
    if bspline_fitting_levels is not None:
        corrector.SetNumberOfFittingLevels(bspline_fitting_levels)
    if shrink_factor and shrink_factor > 1:
        shrink_factors = [shrink_factor] * img.GetDimension()
        img_small = sitk.Shrink(img, shrink_factors)
        mask_small = sitk.Shrink(mask_img, shrink_factors)
        # First pass on smaller image (faster)
        corrected_small = corrector.Execute(img_small, mask_small)
        # Reconstruct bias field at full resolution and correct full image
        log_bias_field = corrector.GetLogBiasFieldAsImage(img)  # full-res bias field
        bias_field = sitk.Exp(-log_bias_field)
        # Ensure both images are float32 before multiplying
        if bias_field.GetPixelID() != sitk.sitkFloat32:
            bias_field = sitk.Cast(bias_field, sitk.sitkFloat32)
        if img.GetPixelID() != sitk.sitkFloat32:
            img = sitk.Cast(img, sitk.sitkFloat32)
        corrected = bias_field * img
    else:
        corrected = corrector.Execute(img, mask_img)
    sitk.WriteImage(corrected, out_path)
    return out_path

# Normalisation
def normalize(
    nifti_path: str,
    out_path: str | None = None,
    mask_path: str | None = None,
    method: str = "zscore",
    clip_percentiles: tuple[float, float] = (2.0, 98.0),
    eps: float = 1e-8,
) -> str:
    """Normalize image intensities into a stable range.

    Supported methods:
        zscore: Standard (x - mean) / std.
        robust_z: Median + MAD based z-score (outlier resistant).
        minmax: Scale brain-region min→0, max→1.
        percentile: Use lower/upper percentiles for scaling bounds.
        zscore_clipped: Clip to percentiles first, then z-score.

    Args:
        nifti_path: Path to input NIfTI image.
        out_path: Optional output path; auto-generated if None.
        mask_path: Optional brain mask path; if absent, all non-zero voxels used.
        method: Normalization strategy name.
        clip_percentiles: (low, high) percentiles for clipping-based methods.
        eps: Numerical stability epsilon.

    Returns:
        str: Path to normalized NIfTI file.
    Raises:
        FileNotFoundError: If input image is missing.
        ValueError: If method is unknown or brain mask is empty.
    """
    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(nifti_path)
    if out_path is None:
        base, ext = os.path.splitext(nifti_path)
        if base.endswith(".nii"):
            base = base[:-4]
        out_path = base + "_norm.nii.gz"

    img_nib = nib.load(nifti_path)
    data = img_nib.get_fdata(dtype=float)

    if mask_path and os.path.isfile(mask_path):
        mask = nib.load(mask_path).get_fdata()
        brain = mask > 0.5
    else:
        # Fallback: use all non-zero voxels
        brain = data != 0

    if not brain.any():
        raise ValueError("Brain mask empty – cannot normalize.")

    region = data[brain]

    low_p, high_p = clip_percentiles
    p_low = np.percentile(region, low_p)
    p_high = np.percentile(region, high_p)

    if method == "zscore":
        m = region.mean()
        s = region.std()
        norm = (data - m) / (s + eps)

    elif method == "robust_z":
        med = np.median(region)
        mad = np.median(np.abs(region - med)) + eps
        # Approximate std from MAD (Gaussian): std ≈ 1.4826 * MAD
        norm = (data - med) / (1.4826 * mad)

    elif method == "minmax":
        mn = region.min()
        mx = region.max()
        norm = (data - mn) / (mx - mn + eps)

    elif method == "percentile":
        norm = (data - p_low) / (p_high - p_low + eps)
        norm = np.clip(norm, 0.0, 1.0)

    elif method == "zscore_clipped":
        clipped = np.clip(region, p_low, p_high)
        m = clipped.mean()
        s = clipped.std()
        norm = (data - m) / (s + eps)
        # Optional clamp to reasonable range
        norm = np.clip(norm, -5, 5)

    else:
        raise ValueError(f"Unknown method '{method}'.")

    # Preserve affine & header
    out_img = nib.Nifti1Image(norm.astype(np.float32), img_nib.affine, img_nib.header)
    nib.save(out_img, out_path)
    return out_path

# Resampling
def resample(
    nifti_path: str,
    out_path: str | None = None,
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    interpolation: str = "linear",
    is_mask: bool = False,
) -> str:
    """Resample an image to desired voxel spacing (e.g. isotropic 1 mm).

    Args:
        nifti_path: Path to input NIfTI image.
        out_path: Optional output path; auto-generated if None.
        target_spacing: Desired spacing (sx, sy, sz) in mm.
        interpolation: Interpolation mode ("linear", "nearest", "bspline").
        is_mask: If True, force nearest-neighbour interpolation.

    Returns:
        str: Path to resampled NIfTI image.
    Raises:
        FileNotFoundError: If input path is missing.
        ValueError: If interpolation mode is unknown.
    """
    if not os.path.isfile(nifti_path):
        raise FileNotFoundError(nifti_path)
    if out_path is None:
        base, ext = os.path.splitext(nifti_path)
        if base.endswith('.nii'):
            base = base[:-4]
        out_path = base + '_res.nii.gz'

    img = sitk.ReadImage(nifti_path)
    orig_spacing = img.GetSpacing()  # (sx, sy, sz)
    orig_size = img.GetSize()        # (nx, ny, nz)
    tgt = tuple(float(s) for s in target_spacing)
    new_size = [int(round(orig_size[i] * orig_spacing[i] / tgt[i])) for i in range(3)]

    if is_mask:
        interp = sitk.sitkNearestNeighbor
    else:
        if interpolation == 'linear':
            interp = sitk.sitkLinear
        elif interpolation == 'nearest':
            interp = sitk.sitkNearestNeighbor
        elif interpolation == 'bspline':
            interp = sitk.sitkBSpline
        else:
            raise ValueError(f"Unknown interpolation '{interpolation}'")

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(tgt)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetInterpolator(interp)
    resampled = resampler.Execute(img)
    sitk.WriteImage(resampled, out_path)
    return out_path
