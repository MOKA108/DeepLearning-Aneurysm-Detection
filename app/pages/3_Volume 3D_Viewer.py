"""Streamlit 3D volume renderer for aneurysm visualization.

Interactive 3D visualization using Plotly Volume rendering with aneurysm localization
markers, focal cube highlighting, and optional cropped views for detailed inspection.

Features:
    - Full volume 3D rendering with adjustable opacity and colorscale
    - Dual coordinate display: predicted (cyan) and ground truth (green)
    - Focal cube (32³) highlighting with red voxel overlay
    - Yellow wireframe marking region of interest
    - Cropped view (32³) around predicted coordinate
    - Adjustable intensity windowing and iso-range limiting
    - Automatic coordinate loading from inference results

Run:
    streamlit run src/app/streamlit_app.py

Data Sources:
    - Volume: data/preprocessed_data/resample/<uid>_res.nii
    - Probability map: src/app/results/<uid>/prob_map.npy
    - Predicted coords: src/app/results/<uid>/visualization_summary.json
    - Ground truth: data/preprocessed_data/mapping/<uid>_mapping.json

Visualization Legend:
    - Green marker (size=10): Ground truth aneurysm location
    - Cyan marker (size=8): Predicted aneurysm coordinate
    - Yellow wireframe: 32³ focal cube boundary
    - Red voxels: High-probability region within focal cube
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import nibabel as nib
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import json
import re
try:
    from scipy import ndimage as _ndi
except Exception:
    _ndi = None

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

def volume_render(
    vol: np.ndarray,
    downsample: int,
    opacity: float,
    surface_count: int,
    colorscale: str,
    values_override: np.ndarray | None = None,
    prob_alpha: np.ndarray | None = None,
    isomin: float | None = None,
    isomax: float | None = None,
    threshold: float | None = None,
    accentuate_threshold: bool = False,
    low_threshold_opacity: float | None = None,
) -> go.Figure:
    """Render a 3D volume with optional value override, probability-based opacity, and iso-range.

    prob_alpha: if provided (same shape), used to build an opacityscale so low probability becomes transparent.
    isomin/isomax: restrict visible value range (if not None).
    """
    # Downsample voxel values but keep original coordinate indices for correct marker placement.
    ds = vol[::downsample, ::downsample, ::downsample]
    if values_override is not None:
        vals = values_override[::downsample, ::downsample, ::downsample]
    else:
        vals = ds

    opacityscale = None
    if prob_alpha is not None and prob_alpha.shape == vol.shape:
        alphas = prob_alpha[::downsample, ::downsample, ::downsample].flatten()
        vmin, vmax = float(vals.min()), float(vals.max())
        rng = vmax - vmin if vmax > vmin else 1.0
        vn = (vals.flatten() - vmin) / (rng + 1e-8)
        opacityscale = []
        # sample 8 points across normalized value range; map to average alpha nearby
        for t in np.linspace(0, 1, 8):
            mask = (vn >= t - 0.05) & (vn <= t + 0.05)
            a = float(alphas[mask].mean()) if mask.any() else float(t)
            a = min(1.0, max(0.02, a))  # clamp
            opacityscale.append([float(t), a])

    # Use original index positions rather than compressed range (avoid shifting markers).
    x_coords = np.arange(0, vol.shape[0], downsample)
    y_coords = np.arange(0, vol.shape[1], downsample)
    z_coords = np.arange(0, vol.shape[2], downsample)
    x, y, z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    # Build custom colorscale & opacityscale if accentuation requested and threshold provided.
    custom_colorscale = None
    if accentuate_threshold and threshold is not None:
        t = float(max(0.0, min(1.0, threshold)))
        # Binary-like visual: below threshold faint gray, above threshold bright red
        custom_colorscale = [
            [0.0, "rgb(70,70,70)"],
            [max(0.0, t - 1e-4), "rgb(80,80,80)"],
            [t, "rgb(255,0,0)"],
            [1.0, "rgb(255,0,0)"],
        ]
        base_low = 0.05 if low_threshold_opacity is None else float(low_threshold_opacity)
        base_low = max(0.01, min(0.3, base_low))
        opacityscale = [
            [0.0, base_low],
            [max(0.0, t - 1e-4), base_low],
            [t, min(1.0, max(0.6, opacity * 4))],
            [1.0, min(1.0, max(0.85, opacity * 5))],
        ]

    kwargs = dict(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        value=vals.flatten(),
        opacity=opacity,
        surface_count=surface_count,
        colorscale=custom_colorscale if custom_colorscale is not None else colorscale,
    )
    if isomin is not None:
        kwargs["isomin"] = isomin
    if isomax is not None:
        kwargs["isomax"] = isomax
    if opacityscale is not None:
        kwargs["opacityscale"] = opacityscale

    fig = go.Figure(data=go.Volume(**kwargs))
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    return fig

def add_point_marker(fig: go.Figure, coord: tuple[int, int, int], color: str = 'red', size: int = 6, name: str = 'Aneurysm') -> None:
    """Add 3D scatter marker to Plotly figure at specified coordinate.
    
    Args:
        fig: Plotly Figure object to modify in-place.
        coord: (x, y, z) coordinate tuple for marker position.
        color: Marker color (CSS color string).
        size: Marker size in pixels.
        name: Legend label for marker trace.
    """
    x, y, z = coord
    fig.add_trace(go.Scatter3d(x=[x], y=[y], z=[z], mode='markers',
                               marker=dict(size=size, color=color, symbol='circle'), name=name))

def safe_crop(vol: np.ndarray, center: tuple[int, int, int], margin: int) -> tuple[np.ndarray, tuple[int, int, int]]:
    x, y, z = center
    x0 = max(x - margin, 0); x1 = min(x + margin + 1, vol.shape[0])
    y0 = max(y - margin, 0); y1 = min(y + margin + 1, vol.shape[1])
    z0 = max(z - margin, 0); z1 = min(z + margin + 1, vol.shape[2])
    cropped = vol[x0:x1, y0:y1, z0:z1]
    return cropped, (x0, y0, z0)

@st.cache_data(show_spinner=False)
def load_inference_json(path: str) -> dict | None:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception:
        return None

st.set_page_config(page_title="Aneurysm 3D Viewer", layout="wide")
st.title("Aneurysm Volume 3D Viewer")

with st.sidebar:
    st.header("Input Files")
    # Limit selection to 5 predefined NIfTI files.
    VOLUME_CHOICES = {
        "Volume 1": "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.10237346404947508483392228545497384153_res.nii",
        "Volume 2": "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.86587095315940322589473858425390856478_res.nii",
        "Volume 3": "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.11467864225381867528457397602560884904_res.nii",
        "Volume 4": "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.12356213647263814629765845148859853799_res.nii",
        "Volume 5": "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.44002135066368305909834635759388782294_res.nii",
    }
    selected_label = st.selectbox("NIfTI Volume", list(VOLUME_CHOICES.keys()), index=0)
    selected_volume = VOLUME_CHOICES[selected_label]
    # Extract UID from filename (before _res.nii or _res.nii.gz)
    def extract_uid(path: str) -> str:
        fname = Path(path).name
        m = re.match(r"(.*?)_res\.nii(\.gz)?$", fname)
        return m.group(1) if m else fname
    series_uid = extract_uid(selected_volume)
    results_root = Path("src/app/results") / series_uid
    prob_map_path_text = str(results_root / "prob_map.npy")
    inference_json_default = str(results_root / "visualization_summary.json")
    mapping_json_path = Path("data/preprocessed_data/mapping") / f"{series_uid}_mapping.json"
    st.markdown(f"Auto prob map path: `{prob_map_path_text}`")
    st.markdown(f"Auto JSON path: `{inference_json_default}`")

    st.header("2D Parameters")
    clip_percent = st.slider("Clip intensity %", 0.0, 5.0, 0.5, 0.1)
    st.markdown("---")
    st.header("3D Parameters")
    win_lo = st.slider("Window low percentile", 0, 20, 1)
    win_hi = st.slider("Window high percentile", 80, 100, 99)
    colorscale = st.selectbox("3D color palette", ["Gray", "Viridis", "Hot", "Turbo", "Magma", "Inferno", "IceFire", "RdBu"], index=0)
    downsample = st.slider("Downsample 3D", 1, 8, 3)
    st.caption(f"Downsample 3D: step={downsample}")
    opacity = st.slider("3D volume opacity", 0.05, 0.5, 0.15, 0.01)
    surface_count = st.slider("3D surface count", 5, 30, 15)
    st.markdown("---")
    st.header("3D Rendering Refinement")
    use_isorange = st.checkbox("Limit range (isomin/isomax)", value=False)
    iso_min = st.slider("isomin", 0.0, 1.0, 0.1, 0.01, disabled=not use_isorange)
    iso_max = st.slider("isomax", 0.0, 1.0, 0.9, 0.01, disabled=not use_isorange)

    st.header("Aneurysm Coordinate")
    inference_json_path = st.text_input("Result JSON path", value=inference_json_default)
    load_coord = st.button("Load coord from JSON")
    # Initialize session state for coordinates
    if 'coord_x' not in st.session_state:
        st.session_state.coord_x = 0
    if 'coord_y' not in st.session_state:
        st.session_state.coord_y = 0
    if 'coord_z' not in st.session_state:
        st.session_state.coord_z = 0
    # Auto-load if JSON exists and coordinates are still (0,0,0)
    if Path(inference_json_default).exists() and all(st.session_state[k] == 0 for k in ['coord_x','coord_y','coord_z']):
        try:
            data_auto = json.load(open(inference_json_default,'r',encoding='utf-8'))
            # Try multiple possible keys for coordinate
            coord_keys = ['predicted_coordinate', 'pred_coord', 'aneurysm_coordinate', 'coordinate']
            for key in coord_keys:
                if key in data_auto and isinstance(data_auto[key], list) and len(data_auto[key]) == 3:
                    cx, cy, cz = [int(v) for v in data_auto[key]]
                    st.session_state.coord_x = cx
                    st.session_state.coord_y = cy
                    st.session_state.coord_z = cz
                    break
            if 'threshold' in data_auto and 'threshold_json' not in st.session_state:
                try:
                    st.session_state.threshold_json = float(data_auto['threshold'])
                except Exception:
                    pass
        except Exception:
            pass
    coord_x = st.number_input("X", min_value=0, value=st.session_state.coord_x, key="coord_x")
    coord_y = st.number_input("Y", min_value=0, value=st.session_state.coord_y, key="coord_y")
    coord_z = st.number_input("Z", min_value=0, value=st.session_state.coord_z, key="coord_z")
    st.markdown("---")
    st.header("Visualization Options")
    # Crop now fixed to edge length 32 (same as cube highlight)
    show_crop = st.checkbox("Show cropped volume (32³ fixed)", value=True)
    highlight_point = st.checkbox("Show 3D point", value=True)
    st.markdown("---")
    st.header("Focal Cube Mode")
    use_cube_highlight = st.checkbox("Show focal cube (overlay)", value=True)
    cube_edge = 32
    st.caption("Cube edge fixed at 32 voxels")

vol_data: np.ndarray | None = None
prob_data: np.ndarray | None = None

try:
    if selected_volume and Path(selected_volume).exists():
        vol_data = load_nifti(selected_volume)
    else:
        st.error("Selected file does not exist.")
except Exception as e:
    st.error(f"Failed to load volume: {e}")

try:
    if prob_map_path_text and Path(prob_map_path_text).exists():
        prob_data = load_prob_map(prob_map_path_text)
except Exception as e:
    st.warning(f"Probability map not loaded: {e}")

if vol_data is None:
    st.warning("No volume loaded.")
    st.stop()

norm_vol = normalize_vol(vol_data, clip_percent=clip_percent)
windowed_vol = window_intensity(vol_data, win_lo, win_hi)
# Base volume for 3D rendering: use normalized volume to keep anatomical structure visible
render_base_vol = norm_vol

# Load ground truth coordinate from mapping JSON if available
gt_coord = None
if mapping_json_path.exists():
    try:
        with open(mapping_json_path, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        # Mapping file is an array, get first element
        if isinstance(mapping_data, list) and len(mapping_data) > 0:
            mapping_entry = mapping_data[0]
            if 'voxel' in mapping_entry and isinstance(mapping_entry['voxel'], list) and len(mapping_entry['voxel']) == 3:
                gt_coord = tuple(int(round(v)) for v in mapping_entry['voxel'])
                st.sidebar.success(f"Ground truth coord loaded: {gt_coord}")
            elif 'aneurysm_coordinate' in mapping_entry:
                gt_coord = tuple(int(round(v)) for v in mapping_entry['aneurysm_coordinate'])
                st.sidebar.success(f"Ground truth coord loaded: {gt_coord}")
    except Exception as e:
        st.sidebar.warning(f"Could not load ground truth: {e}")

# Load coordinate from JSON if requested
json_coord = None
if load_coord and inference_json_path and Path(inference_json_path).exists():
    data = load_inference_json(inference_json_path)
    if data:
        possible_keys = ["predicted_coordinate", "aneurysm_coordinate", "coordinate", "pred_coord"]
        for k in possible_keys:
            if k in data and isinstance(data[k], (list, tuple)) and len(data[k]) == 3:
                try:
                    json_coord = tuple(int(round(v)) for v in data[k])
                except Exception:
                    json_coord = None
                break
        if json_coord is None:
            for v in data.values():
                if isinstance(v, (list, tuple)) and len(v) == 3 and all(isinstance(x, (int, float)) for x in v):
                    json_coord = tuple(int(round(x)) for x in v)
                    break
    if json_coord:
        st.session_state.coord_x, st.session_state.coord_y, st.session_state.coord_z = json_coord
        coord_x, coord_y, coord_z = json_coord
        st.success(f"Coordinate loaded: {json_coord}")
    if 'threshold' in data and isinstance(data['threshold'], (int,float)):
        st.session_state.threshold_json = float(data['threshold'])
    else:
        st.warning("Unable to find (x,y,z) coordinate in JSON.")

aneurysm_coord = (int(coord_x), int(coord_y), int(coord_z))
inside = all(0 <= aneurysm_coord[i] < vol_data.shape[i] for i in range(3))
if not inside:
    st.warning("Coordinate out of volume bounds.")

st.subheader("3D Volume Rendering")
st.markdown("""
**Legend:**
- 🟢 **Green marker**: Ground truth aneurysm location (if available)
- 🔵 **Cyan marker**: Predicted aneurysm coordinate
- 🟡 **Yellow wireframe**: 32³ focal cube around prediction
- 🔴 **Red voxels**: High-probability voxels within risk zone (above threshold)
""")
show_3d = st.button("Show 3D rendering")
if show_3d:
    # Create a modified volume where cube voxels have a distinct value
    values_for_render = render_base_vol.copy()
    cube_mask = None
    if use_cube_highlight and inside:
        half = cube_edge // 2
        x0, y0, z0 = aneurysm_coord
        cx_min = max(x0 - half, 0); cx_max = min(x0 + half + 1, render_base_vol.shape[0])
        cy_min = max(y0 - half, 0); cy_max = min(y0 + half + 1, render_base_vol.shape[1])
        cz_min = max(z0 - half, 0); cz_max = min(z0 + half + 1, render_base_vol.shape[2])
        # Marquer les voxels du cube en ajoutant 10 (pour les distinguer dans le colorscale)
        cube_mask = np.zeros(render_base_vol.shape, dtype=bool)
        cube_mask[cx_min:cx_max, cy_min:cy_max, cz_min:cz_max] = True
        values_for_render = render_base_vol.copy()
        # Normalize so volume is in [0, 0.5] and cube in [0.5, 1.0]
        values_for_render = values_for_render * 0.5  # base volume: 0 to 0.5
        values_for_render[cube_mask] = 0.5 + render_base_vol[cube_mask] * 0.5  # cube: 0.5 to 1.0
        
        # Create custom colorscale: gray for [0, 0.5], red for [0.5, 1.0]
        custom_colorscale = [
            [0.0, "rgb(50,50,50)"],
            [0.49, "rgb(120,120,120)"],
            [0.5, "rgb(255,0,0)"],
            [1.0, "rgb(255,100,100)"]
        ]
    else:
        custom_colorscale = colorscale
    
    # Render with modified volume
    fig = volume_render(
        values_for_render,
        downsample=downsample,
        opacity=opacity,
        surface_count=surface_count,
        colorscale=custom_colorscale if cube_mask is not None else colorscale,
        values_override=None,
        prob_alpha=None,
        isomin=iso_min if use_isorange else None,
        isomax=iso_max if use_isorange else None,
        threshold=None,
        accentuate_threshold=False,
        low_threshold_opacity=None,
    )
    st.caption(f"Shape volume: {vol_data.shape} | Shape prob_map: {prob_data.shape if prob_data is not None else 'None'} | Downsample={downsample}")
    if prob_data is not None and prob_data.shape != vol_data.shape:
        st.error("Mismatch shapes volume/prob_map: overlay 3D partielle (marker OK).")
    
    # Add ground truth marker if available
    if gt_coord is not None:
        gt_inside = all(0 <= gt_coord[i] < vol_data.shape[i] for i in range(3))
        if gt_inside:
            add_point_marker(fig, gt_coord, color='green', size=10, name='Ground Truth')
    
    # Add predicted marker
    if highlight_point and inside:
        add_point_marker(fig, aneurysm_coord, color='cyan', size=8, name='Predicted')
    if use_cube_highlight and inside:
        half = cube_edge // 2
        x0, y0, z0 = aneurysm_coord
        cx_min = max(x0 - half, 0); cx_max = min(x0 + half + 1, render_base_vol.shape[0])
        cy_min = max(y0 - half, 0); cy_max = min(y0 + half + 1, render_base_vol.shape[1])
        cz_min = max(z0 - half, 0); cz_max = min(z0 + half + 1, render_base_vol.shape[2])
        corners = [
            (cx_min, cy_min, cz_min), (cx_min, cy_min, cz_max-1), (cx_min, cy_max-1, cz_min), (cx_min, cy_max-1, cz_max-1),
            (cx_max-1, cy_min, cz_min), (cx_max-1, cy_min, cz_max-1), (cx_max-1, cy_max-1, cz_min), (cx_max-1, cy_max-1, cz_max-1)
        ]
        edges = [
            (0,1),(0,2),(1,3),(2,3), (4,5),(4,6),(5,7),(6,7), (0,4),(1,5),(2,6),(3,7)
        ]
        ex=[]; ey=[]; ez=[]
        for a,b in edges:
            xA,yA,zA = corners[a]; xB,yB,zB = corners[b]
            ex += [xA,xB,None]; ey += [yA,yB,None]; ez += [zA,zB,None]
        fig.add_trace(
            go.Scatter3d(x=ex, y=ey, z=ez, mode='lines', line=dict(color='yellow', width=4), name='Aneurysm Zone')
        )

    st.plotly_chart(fig, use_container_width=True)
    
    # Show cropped volume if enabled
    if show_crop and inside:
        st.subheader("Cropped volume (32³) around coordinate")
        crop_edge = 32
        half = crop_edge // 2
        x0, y0, z0 = aneurysm_coord
        x_min = max(x0 - half, 0); x_max = min(x0 + half + 1, render_base_vol.shape[0])
        y_min = max(y0 - half, 0); y_max = min(y0 + half + 1, render_base_vol.shape[1])
        z_min = max(z0 - half, 0); z_max = min(z0 + half + 1, render_base_vol.shape[2])
        cropped_vol = render_base_vol[x_min:x_max, y_min:y_max, z_min:z_max]
        offset = (x_min, y_min, z_min)
        crop_fig = volume_render(cropped_vol, downsample=1, opacity=opacity, surface_count=max(1, surface_count//2), colorscale=colorscale)
        local_coord = (x0 - x_min, y0 - y_min, z0 - z_min)
        add_point_marker(crop_fig, local_coord, color='cyan', size=5, name='Focus')
        st.plotly_chart(crop_fig, use_container_width=True)
        st.caption(f"Offset: {offset} | Crop shape: {cropped_vol.shape} | Target edge=32")
else:
    st.info("Click 'Show 3D rendering' to generate the volume.")

st.success("Visualization complete.")
