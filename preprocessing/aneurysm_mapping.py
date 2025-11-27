"""Map aneurysm annotation points from 2D DICOM slice coordinates to 3D NIfTI voxel indices.

Overview:
    This module loads aneurysm annotation points stored as (x, y) pixel coordinates on individual
    DICOM slices and converts them into floating-point voxel indices inside a NIfTI volume so
    training code can directly extract 3D patches. DICOM geometry is expressed in patient LPS
    space while nibabel exposes affines in RAS space; a consistent conversion is performed.

Pipeline Summary:
    1. Parse CSV rows -> structured `AneurysmAnnotation` objects.
    2. For each annotation, compute its physical LPS coordinate using ImagePositionPatient (IPP),
       ImageOrientationPatient (IOP) and PixelSpacing.
    3. Convert LPS -> RAS (sign inversion of L and P) and apply the inverse NIfTI affine to obtain
       continuous voxel indices (i, j, k).
    4. Optionally flip left/right and/or remap coordinates into a resampled image space.

Key DICOM Geometry:
    physical_point = IPP
                      + row_index * row_dir * row_spacing
                      + col_index * col_dir * col_spacing
    (row_index = y, col_index = x unless `row_col_swapped=True`).

Edge Cases Managed:
    - Missing reliable slice ordering: fall back from InstanceNumber to ImagePositionPatient.
    - Multi-frame DICOM: optional frame key 'f' retained if present.
    - Optional left/right flip for orientation mismatches: `flip_lr=True`.

Public API (primary helpers):
    map_annotations_to_nifti
    map_annotations_with_slice_index
    map_annotations_to_resampled
    remap_voxel_to_new_image / remap_voxels_to_new_image

The math details (LPS ↔ RAS) can be ignored by callers using the convenience functions.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import ast
import csv

import numpy as np
import pydicom
import nibabel as nib

@dataclass
class AneurysmAnnotation:
    series_uid: str
    sop_uid: str
    x: float
    y: float
    frame: Optional[int]
    location: str

def parse_annotation_row(row: dict) -> AneurysmAnnotation:
    """Parse a CSV dictionary row into an `AneurysmAnnotation`.

    Args:
        row: Dictionary produced by `csv.DictReader` containing annotation fields.

    Returns:
        AneurysmAnnotation: Structured annotation with numeric coordinates.
    """
    coord_str = row["coordinates"]
    d = ast.literal_eval(coord_str)  # trusted source assumption
    return AneurysmAnnotation(
        series_uid=row["SeriesInstanceUID"],
        sop_uid=row["SOPInstanceUID"],
        x=float(d["x"]),
        y=float(d["y"]),
        frame=int(d["f"]) if "f" in d else None,
        location=row.get("location", "")
    )

def load_annotations(csv_path: str, series_uid: Optional[str] = None) -> List[AneurysmAnnotation]:
    """Load all annotations from a CSV, optionally filtering by series UID.

    Args:
        csv_path: Path to annotations CSV file.
        series_uid: If provided, retain only rows matching this `SeriesInstanceUID`.

    Returns:
        List of `AneurysmAnnotation` objects.
    """
    ann: List[AneurysmAnnotation] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if series_uid and r["SeriesInstanceUID"] != series_uid:
                continue
            ann.append(parse_annotation_row(r))
    return ann

def _dicom_slice_lookup(series_dir: str) -> dict:
    """Return a dictionary mapping SOPInstanceUID -> pydicom Dataset for all *.dcm files.

    Args:
        series_dir: Directory containing DICOM slice files.

    Returns:
        Dict[str, pydicom.dataset.FileDataset]: Mapping for quick dataset retrieval.
    """
    out = {}
    for dcm_file in Path(series_dir).glob("*.dcm"):
        ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=False)
        out[ds.SOPInstanceUID] = ds
    return out

def _sorted_sops_by_instance(sop_map: dict) -> List[str]:
    """Order SOPInstanceUIDs by slice sequence.

    Primary key: `InstanceNumber` (if present). Fallback: projection of
    `ImagePositionPatient` on the slice-normal axis.

    Args:
        sop_map: Mapping SOPInstanceUID -> pydicom Dataset.

    Returns:
        Ordered list of SOPInstanceUID strings.
    """
    items = list(sop_map.items())
    # Try by InstanceNumber
    def sort_key(t):
        ds = t[1]
        inst = getattr(ds, 'InstanceNumber', None)
        if inst is not None:
            return ('0', int(inst))  # ensure ints go first grouping
        # fallback: use ImagePositionPatient projection on normal axis if possible
        ipp = getattr(ds, 'ImagePositionPatient', None)
        if ipp is not None:
            try:
                zval = float(ipp[2])  # crude; adequate for ordering
                return ('1', zval)
            except Exception:
                pass
        return ('2', 0)
    items.sort(key=sort_key)
    return [uid for uid, _ in items]

def annotation_to_physical_LPS(annotation: AneurysmAnnotation, ds: pydicom.dataset.FileDataset, row_col_swapped: bool = False) -> np.ndarray:
    """Convert annotation pixel coordinates to a physical (x, y, z) point in LPS space.

    Args:
        annotation: Annotation containing (x, y) pixel coordinate.
        ds: DICOM dataset for the slice.
        row_col_swapped: If True, treat annotation.x as row index and annotation.y as column index.

    Returns:
        np.ndarray: Physical LPS coordinate (float64 shape (3,)).
    """
    ipp = np.array(ds.ImagePositionPatient, dtype=float)  # (x,y,z) LPS origin for top-left pixel
    iop = np.array(ds.ImageOrientationPatient, dtype=float)  # 6 values
    row_dir = iop[0:3]
    col_dir = iop[3:6]
    pix_sp = ds.PixelSpacing  # (row, col)
    row_spacing, col_spacing = float(pix_sp[0]), float(pix_sp[1])
    # DICOM pixel indices are (row=y, col=x). Annotation x,y assumed (x=column, y=row)
    if row_col_swapped:
        # Variant B requested: treat annotation.x as row index, annotation.y as column index
        physical = ipp + annotation.x * row_spacing * row_dir + annotation.y * col_spacing * col_dir
    else:
        physical = ipp + annotation.y * row_spacing * row_dir + annotation.x * col_spacing * col_dir
    return physical  # LPS

def physical_LPS_to_nifti_voxel(physical_lps: np.ndarray, nifti_img: nib.Nifti1Image) -> np.ndarray:
    """Transform a physical LPS coordinate into floating voxel indices.

    Steps:
        1. Convert LPS -> RAS by negating L & P components.
        2. Apply inverse NIfTI affine to obtain continuous voxel indices.

    Args:
        physical_lps: Physical coordinate in LPS space (length-3 array).
        nifti_img: Loaded nibabel NIfTI image providing affine.

    Returns:
        np.ndarray: Floating voxel indices (i, j, k) in source image space.
    """
    ras = np.array([-physical_lps[0], -physical_lps[1], physical_lps[2]], dtype=float)
    aff = nifti_img.affine
    inv_aff = np.linalg.inv(aff)
    voxel_h = inv_aff @ np.array([ras[0], ras[1], ras[2], 1.0])
    return voxel_h[0:3]

def verify_voxel_round_trip(voxel: np.ndarray, nifti_img: nib.Nifti1Image) -> np.ndarray:
    """Project voxel indices back to approximate physical LPS coordinates.

    Useful sanity check to confirm left/right flips and spatial consistency.

    Args:
        voxel: Voxel indices (i, j, k) as an array.
        nifti_img: NIfTI image whose affine defines the voxel->RAS transform.

    Returns:
        np.ndarray: Physical LPS coordinate (length-3 array).
    """
    aff = nifti_img.affine
    ras_phys = aff @ np.array([voxel[0], voxel[1], voxel[2], 1.0])
    ras_phys = ras_phys[:3]
    # RAS -> LPS inverse of earlier: L = -R, P = -A, S = S
    lps = np.array([-ras_phys[0], -ras_phys[1], ras_phys[2]], dtype=float)
    return lps

def map_annotations_to_nifti(csv_path: str, series_dir: str, nifti_path: str, row_col_swapped: bool = False, flip_lr: bool = False) -> List[Tuple[AneurysmAnnotation, np.ndarray]]:
    """Map all annotations for a DICOM series into original NIfTI voxel space.

    Args:
        csv_path: Path to annotations CSV.
        series_dir: Directory containing DICOM slices for the series.
        nifti_path: Path to original NIfTI volume.
        row_col_swapped: Toggle interpretation of annotation axes.
        flip_lr: If True, apply left/right flip in voxel space.

    Returns:
        List[Tuple[AneurysmAnnotation, np.ndarray]]: Each tuple holds the annotation and its
        floating voxel indices (i, j, k) in the original image.
    """
    # 1. Read all DICOM slices for the series (we need geometry from each slice).
    sop_map = _dicom_slice_lookup(series_dir)
    # 2. Load NIfTI volume (target space for training / patch extraction).
    nii = nib.load(nifti_path)
    # 3. Derive series UID from first slice; filter CSV rows to this series.
    first_ds = next(iter(sop_map.values()))
    series_uid = first_ds.SeriesInstanceUID
    annotations = load_annotations(csv_path, series_uid=series_uid)
    out: List[Tuple[AneurysmAnnotation, np.ndarray]] = []
    for a in annotations:
        ds = sop_map.get(a.sop_uid)
        if ds is None:
            continue  # annotation references SOP not present
        physical_lps = annotation_to_physical_LPS(a, ds, row_col_swapped=row_col_swapped)
        voxel = physical_LPS_to_nifti_voxel(physical_lps, nii)
        if flip_lr:
            voxel[0] = (nii.shape[0] - 1) - voxel[0]
        out.append((a, voxel))
    return out

def map_annotations_with_slice_index(csv_path: str, series_dir: str, nifti_path: str, row_col_swapped: bool = False, flip_lr: bool = False) -> List[Tuple[AneurysmAnnotation, np.ndarray, int]]:
    """Map annotations to voxel space and also provide their ordered slice index.

    Slice ordering uses `InstanceNumber` with a fallback to geometric `ImagePositionPatient`.

    Args:
        csv_path: Path to annotations CSV.
        series_dir: Series directory containing DICOM slices.
        nifti_path: Path to original NIfTI volume.
        row_col_swapped: Axis interpretation toggle.
        flip_lr: Optional left/right flip.

    Returns:
        List[Tuple[AneurysmAnnotation, np.ndarray, int]]: Annotation, voxel indices, slice index (0-based).
    """
    sop_map = _dicom_slice_lookup(series_dir)
    ordered_sops = _sorted_sops_by_instance(sop_map)
    nii = nib.load(nifti_path)
    first_ds = next(iter(sop_map.values()))
    series_uid = first_ds.SeriesInstanceUID
    annotations = load_annotations(csv_path, series_uid=series_uid)
    out: List[Tuple[AneurysmAnnotation, np.ndarray, int]] = []
    index_lookup = {sop: idx for idx, sop in enumerate(ordered_sops)}  # SOP -> slice order (0-based)
    for a in annotations:
        ds = sop_map.get(a.sop_uid)
        if ds is None:
            continue
        physical_lps = annotation_to_physical_LPS(a, ds, row_col_swapped=row_col_swapped)
        voxel = physical_LPS_to_nifti_voxel(physical_lps, nii)
        if flip_lr:
            voxel[0] = (nii.shape[0] - 1) - voxel[0]
        slice_index = index_lookup.get(a.sop_uid, -1)
        out.append((a, voxel, slice_index))
    return out

__all__ = [
    'AneurysmAnnotation', 'load_annotations', 'map_annotations_to_nifti',
    'annotation_to_physical_LPS', 'physical_LPS_to_nifti_voxel', 'map_annotations_with_slice_index', 'verify_voxel_round_trip',
    'remap_voxel_to_new_image', 'remap_voxels_to_new_image', 'map_annotations_to_resampled'
]

def remap_voxel_to_new_image(voxel: np.ndarray, src_img: nib.Nifti1Image, dst_img: nib.Nifti1Image) -> np.ndarray:
    """Remap a single voxel coordinate between NIfTI images using physical space.

    Args:
        voxel: Source voxel indices (i, j, k).
        src_img: Source NIfTI image.
        dst_img: Destination NIfTI image.

    Returns:
        np.ndarray: Floating voxel indices in destination image space.
    """
    aff_src = src_img.affine
    phys = aff_src @ np.array([voxel[0], voxel[1], voxel[2], 1.0])
    aff_dst_inv = np.linalg.inv(dst_img.affine)
    new_vox_h = aff_dst_inv @ phys
    return new_vox_h[:3]


def remap_voxels_to_new_image(voxels: Iterable[Iterable[float]], src_img: nib.Nifti1Image, dst_img: nib.Nifti1Image, clip: bool = True) -> np.ndarray:
    """Vectorized remap of multiple voxel coordinates into a destination image space.

    Args:
        voxels: Sequence of (i, j, k) voxel coordinates in the source image.
        src_img: Source NIfTI image.
        dst_img: Destination NIfTI image (e.g. resampled isotropic volume).
        clip: If True, clip each axis to valid range [0, size-1].

    Returns:
        np.ndarray: Shape (N, 3) array of remapped floating voxel indices.
    """
    aff_src = src_img.affine
    aff_dst_inv = np.linalg.inv(dst_img.affine)
    vox_arr = np.asarray(voxels, dtype=float)
    # Homogeneous source voxel coordinates
    ones = np.ones((vox_arr.shape[0], 1), dtype=float)
    src_vox_h = np.concatenate([vox_arr, ones], axis=1)
    # Physical RAS for each voxel
    phys = (aff_src @ src_vox_h.T).T  # shape (N,4)
    # Map to destination voxel space
    dst_vox_h = (aff_dst_inv @ phys.T).T  # shape (N,4)
    dst_vox = dst_vox_h[:, :3]
    if clip:
        shape = np.array(dst_img.shape[:3], dtype=float)
        for ax in range(3):
            dst_vox[:, ax] = np.clip(dst_vox[:, ax], 0.0, shape[ax] - 1.0)
    return dst_vox


def map_annotations_to_resampled(
    csv_path: str,
    series_dir: str,
    original_nifti_path: str,
    resampled_nifti_path: str,
    row_col_swapped: bool = False,
    flip_lr: bool = False,
    round_result: bool = False,
    validate: bool = True
) -> List[Tuple[AneurysmAnnotation, np.ndarray]]:
    """Map annotations directly into a resampled image space (e.g. 1 mm³).

    Performs two stages:
        1. Map annotations into the original image using DICOM geometry.
        2. Remap those voxel coordinates into the resampled image via affine transforms.

    Args:
        csv_path: Path to annotations CSV.
        series_dir: Directory containing source DICOM slices.
        original_nifti_path: Path to original NIfTI volume.
        resampled_nifti_path: Path to resampled NIfTI volume.
        row_col_swapped: Toggle annotation axis interpretation.
        flip_lr: Apply optional left/right flip before remapping.
        round_result: If True, round voxel coordinates to nearest integer.
        validate: If True, warn if any coordinate falls outside destination bounds.

    Returns:
        List[Tuple[AneurysmAnnotation, np.ndarray]]: Each contains the annotation and its voxel
        coordinates in the resampled image (float or int depending on `round_result`).
    """
    # Step 1: Map to original image space
    original_mappings = map_annotations_to_nifti(
        csv_path=csv_path,
        series_dir=series_dir,
        nifti_path=original_nifti_path,
        row_col_swapped=row_col_swapped,
        flip_lr=flip_lr
    )

    # Load images
    orig_img = nib.load(original_nifti_path)
    res_img = nib.load(resampled_nifti_path)

    out: List[Tuple[AneurysmAnnotation, np.ndarray]] = []
    for ann, orig_vox in original_mappings:
        res_vox = remap_voxel_to_new_image(orig_vox, orig_img, res_img)
        if round_result:
            res_vox = np.round(res_vox).astype(int)
        if validate:
            for ax, lim in enumerate(res_img.shape[:3]):
                if res_vox[ax] < 0 or res_vox[ax] >= lim:
                    print(f"[warn] Annotation {ann.sop_uid} axis {ax} out of bounds: {res_vox[ax]:.2f} (limit 0..{lim-1})")
                    break
        out.append((ann, res_vox))
    return out
