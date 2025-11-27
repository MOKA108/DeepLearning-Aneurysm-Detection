"""Streamlit Application - Aneurysm Detection and Visualization

Welcome page for the aneurysm detection application.
Provides navigation and overview of available features.

Run:
    streamlit run src/app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Aneurysm Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Aneurysm Detection and Visualization")

st.markdown("""
## Welcome

This application provides tools for detecting and visualizing cerebral aneurysms in MRA (Magnetic Resonance Angiography) volumes using deep learning.

### Available Tools

Navigate using the sidebar to access different features:

#### 🔬 **Inference**
Run aneurysm detection on preprocessed NIfTI volumes using trained deep learning models.
- Select from predefined volumes
- Configure model architecture (TinyUNet3D, Simple3DCNN)
- Adjust inference parameters (patch size, stride, batch size)
- Save probability maps and results

#### 📊 **Slice Viewer (2D)**
Explore MRA volumes slice-by-slice across anatomical planes.
- Simultaneous view of axial, coronal, and sagittal planes
- Probability overlay visualization with adjustable threshold
- Independent navigation for each plane

#### 🎯 **3D Viewer**
Interactive 3D visualization of volumes with aneurysm localization.
- Full 3D volume rendering with adjustable opacity and colorscale
- Focal cube highlighting around detected aneurysm coordinate
- Cropped view for detailed inspection
- Yellow wireframe marking the region of interest
- Cyan marker at predicted aneurysm location

---

### Quick Start

1. **Run Inference** (if not already done):
   - Go to the "Inference" page
   - Select a volume
   - Click "Run Inference"
   - Wait for results to be saved

2. **Visualize Results**:
   - Go to "Slice Viewer" or "3D Viewer"
   - Select the same volume
   - Explore the probability maps and detected coordinates

---

### Data Organization

Results are automatically saved in `src/app/results/<volume_uid>/`:
- `prob_map.npy`: Voxel-wise probability map
- `visualization_summary.json`: Detection metadata (threshold, coordinates, score)

---

### Technical Details

**Models Supported:**
- TinyUNet3D: Lightweight 3D U-Net architecture
- Simple3DCNN: Basic 3D CNN classifier

**Inference Pipeline:**
- Patch-based sliding window approach
- Overlapping patches for smooth probability maps
- Aggregation strategies: max, mean, p95
- Threshold-based binary classification

**Volume Requirements:**
- Format: NIfTI (.nii or .nii.gz)
- Preprocessing: Resampled to standard spacing
- Normalization: Applied during inference

---

### Navigation Tips

💡 Use the **sidebar** to switch between pages

💡 All pages auto-load results from `src/app/results/<uid>/`

💡 Adjust visualization parameters in real-time using the sidebar controls

---
""")

# Show available volumes
st.subheader("📂 Available Volumes")

VOLUME_INFO = {
    "Volume 1": "1.2.826.0.1.3680043.8.498.10237346404947508483392228545497384153",
    "Volume 2": "1.2.826.0.1.3680043.8.498.97970165518053195797247488050816887286",
    "Volume 3": "1.2.826.0.1.3680043.8.498.11467864225381867528457397602560884904",
    "Volume 4": "1.2.826.0.1.3680043.8.498.12356213647263814629765845148859853799",
    "Volume 5": "1.2.826.0.1.3680043.8.498.44002135066368305909834635759388782294",
}

cols = st.columns(len(VOLUME_INFO))
for idx, (label, uid) in enumerate(VOLUME_INFO.items()):
    with cols[idx]:
        results_dir = Path(f"src/app/results/{uid}")
        has_results = results_dir.exists() and (results_dir / "prob_map.npy").exists()
        
        status = "✅ Results available" if has_results else "⚪ No results yet"
        st.markdown(f"**{label}**")
        st.caption(status)
        if has_results:
            st.caption(f"UID: `{uid[:20]}...`")

st.markdown("---")
st.info("👈 Select a page from the sidebar to get started!")
