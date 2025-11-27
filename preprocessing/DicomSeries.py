"""Lightweight DICOM series loader producing a `(Z, H, W)` NumPy volume and minimal metadata.

Purpose:
    Convert a directory of slice `.dcm` files (or a single multi-frame file) into a clean 3D
    array and collect a small subset of normalized metadata needed downstream.

Load Workflow (`DicomSeries(dir).load()`):
    1. Discover all `*.dcm` files.
    2. Read the first file to infer acquisition plane (axial / coronal / sagittal).
    3. Single file:
         - Multi-frame: pixel data already 3D.
         - Single-slice: expand to shape `(1, H, W)`.
       Multiple files:
         - Read & decompress each slice.
         - Sort by `InstanceNumber`; fallback to `ImagePositionPatient` if needed.
         - Stack into a `(num_slices, H, W)` volume.
    4. Extract selected fields (see `COLS_NORM`) into a normalized metadata dict.

Design Notes:
    This intentionally avoids heavyweight DICOM frameworks for transparency and fewer surprises.

Edge Handling:
    - Sorting fallback ensures usable ordering when `InstanceNumber` is absent.
    - Multi-value numeric tags (e.g. `PixelSpacing`) converted to tuples of float.

Primary Outputs:
    - `volume`: 3D NumPy array `(Z, H, W)`.
    - `info.metadata`: Dictionary of normalized selected DICOM tags.
"""

import os
from glob import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import pydicom
from pydicom.multival import MultiValue
from pydicom.dataset import FileDataset

# DICOM tag columns of interest for metadata normalization
COLS_NORM: Tuple[str, ...] = (
    'SeriesInstanceUID',
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 
    'HighBit', 'ImageOrientationPatient', 'ImagePositionPatient', 
    'InstanceNumber', 'Modality', 'PatientID', 
    'PerFrameFunctionalGroupsSequence', 'PhotometricInterpretation', 
    'PixelRepresentation', 'PixelSpacing', 'PlanarConfiguration', 
    'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 
    'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness', 
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID',
    'SeriesInstanceUID'
)

# Data classes / container types for exposing series information & volume
@dataclass(frozen=True)
class SeriesInfo:
    """Succinct metadata describing the loaded DICOM series.

    Attributes:
        plane: Acquisition plane string ("sagittal", "coronal", "axial").
        dim: Integer index of dominant axis normal (0, 1, or 2).
        file_count: Number of `.dcm` files contributing to the volume.
        is_multiframe: True if a single multi-frame file produced the volume.
        metadata: Normalized dictionary of selected DICOM tags.
    """
    # OUI, plane et dim représentent la même chose au final
    plane: str                  # "sagittal", "coronal" ou "axial"
    dim: int                    # 0, 1 ou 2
    file_count: int             # nb de fichiers .dcm
    is_multiframe: bool         # True si unique image multi-frame
    metadata: Dict[str, Any]    # dictionnaire {colonne DICOM normalisée -> valeur}

class DicomSeries:
    """Container for a DICOM series (folder or single file) exposing 3D volume + metadata.

    Sorting priority: `InstanceNumber` → `ImagePositionPatient` fallback.
    Supports multi-frame (already 3D) and single-slice expansion.
    """

    PLANE_MAP = {0: "sagittal", 1: "coronal", 2: "axial"}

    def __init__(self, dir_path: str):
        self.dir_path: str = dir_path
        self._files: List[str] = []
        self._volume: Optional[np.ndarray] = None
        self._info: Optional[SeriesInfo] = None

    def load(self) -> "DicomSeries":
        """Load the series, perform slice ordering, populate volume & `SeriesInfo`.

        Returns:
            DicomSeries: Self for chaining.
        Raises:
            ValueError: If no DICOM files are found or slices cannot be sorted.
        """
        self._files = sorted(glob(os.path.join(self.dir_path, "*.dcm")))
        if not self._files:
            raise ValueError(f"Aucun fichier DICOM trouvé dans {self.dir_path}")

        first = pydicom.dcmread(self._files[0])
        first.decompress()

        # Orientation (which plane the acquisition corresponds to)
        dim = self._infer_plane_from_metadata(first)
        plane = self.PLANE_MAP[dim]

        # Normalized metadata (single reference slice)
        metadata = self._extract_metadata(first)

        # Single file cases: multi-frame vs single-slice
        if len(self._files) == 1:
            if hasattr(first, "NumberOfFrames"):  # multi-frame
                volume = first.pixel_array
                is_multiframe = True
            else:  # single-slice
                volume = np.expand_dims(first.pixel_array, axis=0)
                is_multiframe = False

            self._volume = np.asarray(volume)
            self._info = SeriesInfo(
                plane=plane, dim=dim, file_count=1, is_multiframe=is_multiframe,
                metadata=metadata
            )
            return self

        # Multiple files: read & sort slices, then stack into (Z,H,W)
        raw_slices = [pydicom.dcmread(f) for f in self._files]
        slices = []
        for s in raw_slices:
            s.decompress()
            slices.append(s)

        try:  # primary sorting by InstanceNumber
            slices.sort(key=lambda s: int(s.InstanceNumber))
        except (AttributeError, ValueError):
            try:  # fallback: geometry position along dominant axis
                slices.sort(key=lambda s: float(s.ImagePositionPatient[dim]))
            except Exception as e:
                raise ValueError(
                    "Impossible de trier les coupes : ni InstanceNumber valide, "
                    "ni ImagePositionPatient exploitable."
                ) from e

        volume = np.stack([s.pixel_array for s in slices], axis=0)

        self._volume = np.asarray(volume)
        self._info = SeriesInfo(
            plane=plane, dim=dim,
            file_count=len(self._files), is_multiframe=False,
            metadata=metadata
        )
        return self

    @property
    def volume(self) -> np.ndarray:
        """Return the 3D volume array shaped `(Z, H, W)`.

        Raises:
            RuntimeError: If `load()` has not been called.
        """
        self._ensure_loaded()
        return self._volume  # type: ignore[return-value]

    @property
    def info(self) -> SeriesInfo:
        """Return `SeriesInfo` describing geometric and tag metadata.

        Raises:
            RuntimeError: If `load()` has not been called.
        """
        self._ensure_loaded()
        return self._info  # type: ignore[return-value]

    def as_uint16(self) -> np.ndarray:
        """Cast volume to `uint16` without intensity scaling.

        Returns:
            np.ndarray: Volume as `uint16`.
        """
        v = self.volume
        if np.issubdtype(v.dtype, np.floating):
            v = np.clip(v, 0, 65535)
        return v.astype(np.uint16, copy=False)

    def get_slice(self, index: int) -> np.ndarray:
        """Return a single slice by integer index (supports negatives).

        Args:
            index: Slice index (can be negative referencing from end).
        Returns:
            np.ndarray: 2D slice array.
        """
        v = self.volume
        return v[index]

    def _ensure_loaded(self) -> None:
        if self._volume is None or self._info is None:
            raise RuntimeError("Series not loaded. Call .load() first.")

    @staticmethod
    def _to_tuple_of_float(value: Any) -> Optional[Tuple[float, ...]]:
        """Convert a multi-value DICOM field into a tuple of floats.

        Args:
            value: Raw value (MultiValue / list / tuple / scalar).
        Returns:
            Optional[Tuple[float, ...]]: Tuple of floats or None if conversion fails.
        """
        if value is None:
            return None
        if isinstance(value, (list, tuple, MultiValue)):
            out: List[float] = []
            for v in value:
                try:
                    out.append(float(v))
                except Exception:
                    return None
            return tuple(out) if out else None
        try:
            return (float(value),)
        except Exception:
            return None

    @staticmethod
    def _extract_metadata(ds) -> Dict[str, Any]:
        """Extract selected DICOM tags into a normalized dictionary.

        Numeric multi-value fields (e.g. `PixelSpacing`) become `tuple[float, ...]`.
        Scalar primitive types retained; complex objects stringified.

        Args:
            ds: Pydicom dataset providing tag values.
        Returns:
            Dict[str, Any]: Normalized metadata mapping.
        """
        def get(tag: str, numeric_multi: bool = False) -> Any:
            val = getattr(ds, tag, None)
            if numeric_multi:
                return DicomSeries._to_tuple_of_float(val)
            if val is None:
                return None
            if isinstance(val, (str, int, float, bool)):
                return val
            try:
                return int(val)
            except Exception:
                pass
            try:
                return float(val)
            except Exception:
                pass
            return str(val)

        md: Dict[str, Any] = {}
        for col in COLS_NORM:
            if col in ("PixelSpacing", "WindowCenter", "WindowWidth"):
                md[col] = get(col, numeric_multi=True)
            else:
                md[col] = get(col, numeric_multi=False)
        return md

    @staticmethod
    def _infer_plane_from_metadata(dicom_dataset: FileDataset) -> int:
        """Infer dominant normal axis (0 sagittal, 1 coronal, 2 axial) from orientation vectors.

        Args:
            dicom_dataset: DICOM dataset holding orientation information.
        Returns:
            int: Axis index of acquisition plane normal.
        """
        coord = getattr(dicom_dataset, "ImageOrientationPatient", None)
        if coord is None:
            coord = []
            group_seq = getattr(dicom_dataset, "PerFrameFunctionalGroupsSequence", None)
            plane_pos_seq_min = group_seq[0]["PlanePositionSequence"]
            plane_pos_seq_max = group_seq[-1]["PlanePositionSequence"]
            pos_min = plane_pos_seq_min[0]["ImagePositionPatient"]
            pos_max = plane_pos_seq_max[0]["ImagePositionPatient"]
            for a in pos_min:
                coord.append(float(a))
            for a in pos_max:
                coord.append(float(a))
        row = np.array(coord[0:3], dtype=float)
        col = np.array(coord[3:6], dtype=float)
        normal = np.cross(row, col)
        dim = int(np.argmax(np.abs(normal)))
        return dim
