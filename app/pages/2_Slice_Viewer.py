"""Streamlit 2D slice viewer for anatomical plane navigation.

Provides synchronized multi-plane visualization of MRA volumes with aneurysm
localization overlays. Displays predicted and ground truth coordinates with
focal region highlighting.

Features:
    - Simultaneous axial, coronal, sagittal slice display
    - Focal box (32³) visualization around predicted aneurysm
    - Ground truth coordinate overlay from mapping JSON
    - Independent plane navigation with coordinated sliders
    - Adjustable intensity windowing and clipping
    - Auto-initialization to aneurysm coordinates

Run:
    streamlit run src/app/streamlit_app.py

Data Sources:
    - Volume: data/preprocessed_data/resample/<uid>_res.nii
    - Probability map: src/app/results/<uid>/prob_map.npy
    - Predicted coords: src/app/results/<uid>/visualization_summary.json
    - Ground truth: data/preprocessed_data/mapping/<uid>_mapping.json

Visualization Legend:
    - Green crosshair: Ground truth aneurysm location
    - Cyan crosshair: Predicted aneurysm coordinate
    - Yellow box: 32³ focal region around prediction
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import nibabel as nib
import streamlit as st
import json
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

EPS = 1e-8

@st.cache_data(show_spinner=False)
def load_nifti(path: str) -> np.ndarray:
    """Load NIfTI volume from file path.
    
    Args:
        path: File path to NIfTI volume (.nii / .nii.gz).
    
    Returns:
        np.ndarray: Volume data as float32 array.
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data

@st.cache_data(show_spinner=False)
def load_prob_map(path: str) -> np.ndarray:
    """Load probability map from NumPy .npy file.
    
    Args:
        path: File path to probability map (.npy).
    
    Returns:
        np.ndarray: Probability values as float32 array.
    """
    data = np.load(path).astype(np.float32)
    return data

def normalize_vol(vol: np.ndarray, clip_percent: float | None = 0.5) -> np.ndarray:
    """Normalize volume intensities to [0, 1] range with optional clipping.
    
    Args:
        vol: Input volume array.
        clip_percent: Percentile for intensity clipping (None to disable).
    
    Returns:
        np.ndarray: Normalized volume with values in [0, 1].
    """
    v = vol.copy()
    if clip_percent:
        lo = np.percentile(v, clip_percent)
        hi = np.percentile(v, 100 - clip_percent)
        v = np.clip(v, lo, hi)
    v -= v.min()
    rng = v.max() - v.min() + EPS
    v /= rng
    return v

def window_intensity(vol: np.ndarray, low_perc: float, high_perc: float) -> np.ndarray:
    """Apply percentile-based intensity windowing and rescale to [0, 1].
    
    Args:
        vol: Input volume array.
        low_perc: Lower percentile for window minimum.
        high_perc: Upper percentile for window maximum.
    
    Returns:
        np.ndarray: Windowed and rescaled volume.
    """
    lo = np.percentile(vol, low_perc)
    hi = np.percentile(vol, high_perc)
    if hi - lo < EPS:
        return np.zeros_like(vol)
    v = np.clip(vol, lo, hi)
    v = (v - lo) / (hi - lo + EPS)
    return v.astype(np.float32)

def get_slice(vol: np.ndarray, plane: str, index: int) -> np.ndarray:
    """Extract 2D slice from 3D volume along specified anatomical plane.
    
    Args:
        vol: 3D volume array.
        plane: Anatomical plane ('axial', 'coronal', 'sagittal').
        index: Slice index along plane axis.
    
    Returns:
        np.ndarray: 2D slice array.
    
    Raises:
        ValueError: If plane name is invalid.
    """
    if plane == "axial":
        return vol[index, :, :]
    if plane == "coronal":
        return vol[:, index, :]
    if plane == "sagittal":
        return vol[:, :, index]
    raise ValueError("Invalid plane")

def blend_overlay(base: np.ndarray, overlay: np.ndarray, threshold: float, alpha: float) -> np.ndarray:
    b = np.stack([base, base, base], axis=-1)
    mask = overlay >= threshold
    color = np.zeros_like(b)
    color[..., 0] = overlay  # red channel
    out = b.copy()
    out[mask] = (1 - alpha) * b[mask] + alpha * color[mask]
    return np.clip(out, 0, 1)

def draw_box_on_slice(slice_img: np.ndarray, coord: tuple[int, int, int], plane: str, plane_idx: int, box_edge: int = 32, gt_coord: tuple[int, int, int] | None = None) -> np.ndarray:
    """Draw a bounding box on a slice if the aneurysm coordinate intersects with this slice plane.
    
    Also draws ground truth coordinate if provided.
    
    Returns a matplotlib figure rendered as RGB array with same dimensions as input slice.
    """
    # Get slice dimensions to create figure with matching aspect ratio
    h, w = slice_img.shape
    dpi = 100
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi)
    ax.imshow(slice_img, cmap='gray', interpolation='nearest', aspect='equal')
    ax.axis('off')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Invert y-axis to match image coordinates
    
    x, y, z = coord
    half = box_edge // 2
    
    # Determine if this plane slice intersects the box
    if plane == "axial":
        # Axial slice at plane_idx (x dimension)
        if x - half <= plane_idx <= x + half:
            # Box in y-z plane
            y_min, y_max = max(y - half, 0), min(y + half, slice_img.shape[0])
            z_min, z_max = max(z - half, 0), min(z + half, slice_img.shape[1])
            width = z_max - z_min
            height = y_max - y_min
            rect = Rectangle((z_min, y_min), width, height, linewidth=2, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
            # Draw center point
            ax.plot(z, y, 'c+', markersize=12, markeredgewidth=2)
    
    elif plane == "coronal":
        # Coronal slice at plane_idx (y dimension)
        if y - half <= plane_idx <= y + half:
            # Box in x-z plane
            x_min, x_max = max(x - half, 0), min(x + half, slice_img.shape[0])
            z_min, z_max = max(z - half, 0), min(z + half, slice_img.shape[1])
            width = z_max - z_min
            height = x_max - x_min
            rect = Rectangle((z_min, x_min), width, height, linewidth=2, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
            ax.plot(z, x, 'c+', markersize=12, markeredgewidth=2)
    
    elif plane == "sagittal":
        # Sagittal slice at plane_idx (z dimension)
        if z - half <= plane_idx <= z + half:
            # Box in x-y plane
            x_min, x_max = max(x - half, 0), min(x + half, slice_img.shape[0])
            y_min, y_max = max(y - half, 0), min(y + half, slice_img.shape[1])
            width = y_max - y_min
            height = x_max - x_min
            rect = Rectangle((y_min, x_min), width, height, linewidth=2, edgecolor='yellow', facecolor='none')
            ax.add_patch(rect)
            ax.plot(y, x, 'c+', markersize=12, markeredgewidth=2)
    
    # Draw ground truth point if available
    if gt_coord is not None:
        gt_x, gt_y, gt_z = gt_coord
        if plane == "axial":
            ax.plot(gt_z, gt_y, 'g+', markersize=14, markeredgewidth=2.5, label='Ground Truth')
        elif plane == "coronal":
            ax.plot(gt_z, gt_x, 'g+', markersize=14, markeredgewidth=2.5, label='Ground Truth')
        elif plane == "sagittal":
            ax.plot(gt_y, gt_x, 'g+', markersize=14, markeredgewidth=2.5, label='Ground Truth')
    
    # Convert figure to RGB array
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    fig.canvas.draw()
    img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img_array = img_array[:, :, :3]  # Drop alpha channel
    plt.close(fig)
    
    return img_array / 255.0  # Normalize to [0, 1]

@st.cache_data(show_spinner=False)
def load_inference_json(path: str) -> dict | None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception:
        return None

def extract_uid(path: str) -> str:
    """Extract volume UID from filename by removing _res.nii suffix.
    
    Args:
        path: File path or filename.
    
    Returns:
        str: Extracted UID or original filename if pattern not matched.
    """
    fname = Path(path).name
    m = re.match(r"(.*?)_res\.nii(\.gz)?$", fname)
    return m.group(1) if m else fname

st.set_page_config(page_title="2D Slice Viewer", layout="wide")
st.title("2D Visualization - Three Anatomical Planes")

with st.sidebar:
    st.header("Input Files")
    VOLUME_CHOICES = {
        "Volume 1": "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.10237346404947508483392228545497384153_res.nii",
        "Volume 2": "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.86587095315940322589473858425390856478_res.nii",
        "Volume 3": "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.11467864225381867528457397602560884904_res.nii",
        "Volume 4": "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.12356213647263814629765845148859853799_res.nii",
        "Volume 5": "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.44002135066368305909834635759388782294_res.nii",
    }
    selected_label = st.selectbox("NIfTI Volume", list(VOLUME_CHOICES.keys()), index=0)
    selected_volume = VOLUME_CHOICES[selected_label]
    series_uid = extract_uid(selected_volume)
    results_root = Path("src/app/results") / series_uid
    prob_map_path_text = str(results_root / "prob_map.npy")
    inference_json_default = str(results_root / "visualization_summary.json")
    
    st.header("2D Parameters")
    clip_percent = st.slider("Clip intensity %", 0.0, 5.0, 0.5, 0.1)
    overlay_alpha = st.slider("Overlay alpha", 0.05, 1.0, 0.5, 0.05)
    apply_window = st.checkbox("Apply windowing", value=False)
    win_lo = st.slider("Window low percentile", 0, 20, 1, disabled=not apply_window)
    win_hi = st.slider("Window high percentile", 80, 100, 99, disabled=not apply_window)

# Load volume
vol_data: np.ndarray | None = None
prob_data: np.ndarray | None = None

try:
    if selected_volume and Path(selected_volume).exists():
        vol_data = load_nifti(selected_volume)
    else:
        st.error("Selected file does not exist.")
except Exception as e:
    st.error(f"Failed to load volume: {e}")

# Load probability map
try:
    if prob_map_path_text and Path(prob_map_path_text).exists():
        prob_data = load_prob_map(prob_map_path_text)
except Exception as e:
    st.warning(f"Probability map not loaded: {e}")

if vol_data is None:
    st.warning("No volume loaded.")
    st.stop()

# Load threshold from JSON
prob_threshold = 0.85  # default
aneurysm_coord = None
gt_coord = None

if Path(inference_json_default).exists():
    json_data = load_inference_json(inference_json_default)
    if json_data and 'threshold' in json_data:
        try:
            prob_threshold = float(json_data['threshold'])
            st.sidebar.success(f"Threshold loaded from JSON: {prob_threshold:.3f}")
        except Exception:
            pass
    # Load aneurysm coordinate
    if json_data:
        possible_keys = ["predicted_coordinate", "aneurysm_coordinate", "coordinate", "pred_coord"]
        for k in possible_keys:
            if k in json_data and isinstance(json_data[k], (list, tuple)) and len(json_data[k]) == 3:
                try:
                    aneurysm_coord = tuple(int(round(v)) for v in json_data[k])
                    st.sidebar.info(f"Predicted coord: {aneurysm_coord}")
                except Exception:
                    pass
                break

# Load ground truth coordinate from mapping JSON
mapping_json_path = Path("data/preprocessed_data/mapping") / f"{series_uid}_mapping.json"
if mapping_json_path.exists():
    try:
        with open(mapping_json_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        # Mapping file is an array, get first element
        if isinstance(mapping_data, list) and len(mapping_data) > 0:
            mapping_entry = mapping_data[0]
            if 'voxel' in mapping_entry and isinstance(mapping_entry['voxel'], list) and len(mapping_entry['voxel']) == 3:
                gt_coord = tuple(int(round(v)) for v in mapping_entry['voxel'])
                st.sidebar.success(f"Ground truth coord: {gt_coord}")
            elif 'aneurysm_coordinate' in mapping_entry:
                gt_coord = tuple(int(round(v)) for v in mapping_entry['aneurysm_coordinate'])
                st.sidebar.success(f"Ground truth coord: {gt_coord}")
    except Exception as e:
        st.sidebar.warning(f"Could not load ground truth: {e}")

norm_vol = normalize_vol(vol_data, clip_percent=clip_percent)
if apply_window:
    slice_source_vol = window_intensity(vol_data, win_lo, win_hi)
else:
    slice_source_vol = norm_vol

if prob_data is not None and prob_data.shape != norm_vol.shape:
    st.warning("Probability map shape mismatch; overlay disabled.")
    prob_data = None

st.subheader("Navigation by Planes")
st.markdown("""
**Legend:**
- 🟢 **Green crosshair**: Ground truth aneurysm location (if available)
- 🔵 **Cyan crosshair**: Predicted aneurysm coordinate
- 🟡 **Yellow box**: 32³ focal region around prediction
""")

# Set initial slider values to aneurysm coordinate if available, otherwise use middle
default_axial = aneurysm_coord[0] if aneurysm_coord else norm_vol.shape[0] // 2
default_coronal = aneurysm_coord[1] if aneurysm_coord else norm_vol.shape[1] // 2
default_sagittal = aneurysm_coord[2] if aneurysm_coord else norm_vol.shape[2] // 2

# Sliders for each plane
col1, col2, col3 = st.columns(3)
with col1:
    axial_idx = st.slider("Axial Index", 0, norm_vol.shape[0] - 1, default_axial)
with col2:
    coronal_idx = st.slider("Coronal Index", 0, norm_vol.shape[1] - 1, default_coronal)
with col3:
    sagittal_idx = st.slider("Sagittal Index", 0, norm_vol.shape[2] - 1, default_sagittal)

st.markdown("---")

# Display three planes
planes = [
    ("Axial", "axial", axial_idx),
    ("Coronal", "coronal", coronal_idx),
    ("Sagittal", "sagittal", sagittal_idx)
]

for plane_name, plane_key, idx in planes:
    st.subheader(f"{plane_name} Plane")
    base_slice = get_slice(slice_source_vol, plane_key, idx)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption(f"{plane_name} - Base slice")
        st.image(base_slice, clamp=True)
    
    with col_b:
        if aneurysm_coord is not None:
            # Draw box around aneurysm coordinate instead of probability overlay
            slice_with_box = draw_box_on_slice(base_slice, aneurysm_coord, plane_key, idx, box_edge=32, gt_coord=gt_coord)
            caption_text = f"{plane_name} - Predicted (cyan) "
            if gt_coord is not None:
                caption_text += "+ Ground Truth (green)"
            st.caption(caption_text)
            st.image(slice_with_box, clamp=True)
        else:
            st.caption(f"{plane_name} - No coordinate available")
            st.image(base_slice, clamp=True)
    
    st.markdown("---")

st.success("2D visualization complete.")
