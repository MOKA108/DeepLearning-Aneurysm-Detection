"""Streamlit inference page for aneurysm detection.

Provides interactive interface for running patch-based deep learning inference
on preprocessed MRA volumes. Generates voxel-level probability maps and scan-level
predictions saved to results directory.

Features:
    - Volume selection from predefined dataset
    - Model architecture configuration (TinyUNet3D / Simple3DCNN)
    - Adjustable inference parameters (patch size, stride, batch size, threshold)
    - Real-time progress tracking with time estimates
    - Automatic result persistence (probability maps + metadata JSON)

Run:
    streamlit run src/app/streamlit_app.py

Outputs:
    Results saved to src/app/results/<volume_uid>/:
    - prob_map.npy: Voxel-wise probability map (float32 array)
    - visualization_summary.json: Metadata including predicted coordinate, score, threshold
"""
from __future__ import annotations

import os
import sys
import re
from pathlib import Path
import numpy as np
import nibabel as nib
import streamlit as st
import torch

# Add src path to import model modules
ROOT = Path(__file__).resolve().parent.parent.parent  # Goes to src/
MODEL_PATH = ROOT / "model"
if str(MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(MODEL_PATH))

from aneurysm.model import AneurysmSimple3DCNN, AneurysmTinyUNet3D
from aneurysm.patch_generator import PatchGenerator
from aneurysm.scan_inference import ScanInferenceEngine
from aneurysm.scan_level_evaluator import ScanLevelEvaluator

st.title("Inference - Aneurysm Detection")

VOLUME_CHOICES = [
    "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.10237346404947508483392228545497384153_res.nii",
    "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.86587095315940322589473858425390856478_res.nii",
    "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.11467864225381867528457397602560884904_res.nii",
    "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.12356213647263814629765845148859853799_res.nii",
    "data/preprocessed_data/resample/1.2.826.0.1.3680043.8.498.44002135066368305909834635759388782294_res.nii",
]

selected_volume = st.selectbox("NIfTI Volume", VOLUME_CHOICES, index=0)

def extract_uid(path: str) -> str:
    fname = Path(path).name
    m = re.match(r"(.*?)_res\.nii(\.gz)?$", fname)
    return m.group(1) if m else fname

series_uid = extract_uid(selected_volume)
results_root = Path("src/app/results") / series_uid

st.markdown(f"**Series UID:** `{series_uid}`")
st.markdown(f"**Output folder:** `{results_root}`")

colA, colB = st.columns(2)
with colA:
    model_type = st.selectbox("Architecture", ["tinyunet3d", "simple3dcnn"], index=0)
    base_filters = st.number_input("TinyUNet base filters", min_value=4, max_value=64, value=8, step=4)
    device_choice = st.selectbox("Device", ["auto", "cpu", "cuda"], index=0)
with colB:
    patch_size = st.number_input("Patch size", min_value=16, max_value=96, value=32, step=16)
    stride = st.number_input("Stride", min_value=4, max_value=32, value=10, step=2)
    batch_size = st.number_input("Batch size", min_value=8, max_value=256, value=64, step=8)

agg = st.selectbox("Aggregation", ["max", "mean", "p95"], index=0)
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.85, 0.01)
save_npy = st.checkbox("Save prob_map.npy", value=True)
save_nifti = st.checkbox("Save prob_map.nii.gz", value=False)

weights_path = st.text_input("Weights path (.pth)", value="src/app/best_model.pth")
running_placeholder = st.empty()

infer_button = st.button("Run Inference", type="primary")

@st.cache_data(show_spinner=False)
def load_volume(path: str) -> np.ndarray:
    img = nib.load(path)
    return img.get_fdata().astype(np.float32), img.affine

@st.cache_data(show_spinner=False)
def normalize(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = np.percentile(vol, (1, 99))
    v = np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return v.astype(np.float32)

if infer_button:
    if not Path(selected_volume).exists():
        st.error("Volume not found.")
    elif not Path(weights_path).exists():
        st.error("Weights file not found.")
    else:
        try:
            with st.spinner("Inference in progress..."):
                vol, affine = load_volume(selected_volume)
                norm_vol = normalize(vol)
                mask = (norm_vol > 0).astype(np.uint8)
                if model_type == "simple3dcnn":
                    model = AneurysmSimple3DCNN()
                else:
                    model = AneurysmTinyUNet3D(base_filters=int(base_filters))

                ckpt = torch.load(weights_path, map_location="cpu")
                state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                model.load_state_dict(state, strict=False)

                dev = (
                    torch.device("cuda") if device_choice == "auto" and torch.cuda.is_available() else
                    torch.device(device_choice if device_choice != "auto" else "cpu")
                )
                model.to(dev)
                model.eval()

                patch_gen = PatchGenerator(patch_size=int(patch_size), stride=int(stride))
                coords = patch_gen.generate_from_mask(mask)
                if not coords:
                    st.error("No patch coordinates generated (check stride / patch size).")
                    st.stop()

                engine = ScanInferenceEngine(model, device=dev, batch_size=int(batch_size))
                prob_map = engine.infer_scan(norm_vol, coords, patch_gen)

                evaluator = ScanLevelEvaluator(threshold=float(threshold), aggregation=agg)
                pred_label, score, pred_coord = evaluator.predict_scan(prob_map)

                # Prepare output folder
                results_root.mkdir(parents=True, exist_ok=True)
                if save_npy:
                    np.save(results_root / "prob_map.npy", prob_map.astype(np.float32))
                if save_nifti:
                    import nibabel as nib  # local import
                    prob_img = nib.Nifti1Image(prob_map.astype(np.float32), affine)
                    nib.save(prob_img, results_root / "prob_map.nii.gz")

                summary = {
                    "volume": selected_volume,
                    "uid": series_uid,
                    "weights": weights_path,
                    "model_type": model_type,
                    "tinyunet_base_filters": int(base_filters) if model_type == "tinyunet3d" else None,
                    "patch_size": int(patch_size),
                    "stride": int(stride),
                    "aggregation": agg,
                    "threshold": float(threshold),
                    "score": float(score),
                    "pred_label": int(pred_label),
                    "predicted_coordinate": [int(c) for c in pred_coord],
                    "device": str(dev),
                }
                with open(results_root / "visualization_summary.json", "w", encoding="utf-8") as f:
                    import json
                    json.dump(summary, f, indent=2)
            st.success(f"Complete: label={pred_label} score={score:.4f} coord={pred_coord}")
            st.json(summary)
            st.markdown(f"**Open visualization:** Run `streamlit run src/app/streamlit_app.py` and select the volume for UID `{series_uid}`.")
        except Exception as e:
            st.exception(e)
else:
    st.info("Configure parameters then run inference.")
