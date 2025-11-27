"""Visualize aneurysm inference by overlaying probability map on volume slices.

Generates:
  1. Probability map (voxel-level) from patch-based model.
  2. Overlay PNGs for axial / coronal / sagittal mid-slices.
  3. Optional threshold mask PNGs.
  4. Optional saved NIfTI (.nii.gz) or NumPy (.npy) of probability map.

Example (PowerShell):
  python .\src\app\visualize_inference.py `
    --volume "C:\path\to\scan_res.nii.gz" `
    --weights .\src\app\best_model.pth `
    --model-type tinyunet3d `
    --tinyunet-base-filters 8 `
    --patch-size 32 `
    --stride 10 `
    --aggregation max `
    --threshold 0.85 `
    --out-dir .\viz_out

Outputs saved into out-dir (created if missing).
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import re
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

# Import aneurysm modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from aneurysm.model import AneurysmSimple3DCNN, AneurysmTinyUNet3D
from aneurysm.patch_generator import PatchGenerator
from aneurysm.scan_inference import ScanInferenceEngine
from aneurysm.scan_level_evaluator import ScanLevelEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="Visualize aneurysm probability map")
    p.add_argument("--volume", required=True, help="Path to resampled NIfTI volume (.nii/.nii.gz)")
    p.add_argument("--weights", default="best_model.pth", help="Path to model weights (.pth)")
    p.add_argument("--model-type", default="tinyunet3d", choices=["simple3dcnn", "tinyunet3d"], help="Model architecture")
    p.add_argument("--tinyunet-base-filters", type=int, default=8, help="Base filters for TinyUNet")
    p.add_argument("--patch-size", type=int, default=32, help="Patch edge length")
    p.add_argument("--stride", type=int, default=10, help="Stride for patch grid")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for inference")
    p.add_argument("--aggregation", type=str, default="max", choices=["max", "mean", "p95"], help="Aggregation strategy")
    p.add_argument("--threshold", type=float, default=0.85, help="Decision threshold")
    p.add_argument("--out-dir", type=str, default=None, help="Directory to store outputs (auto if omitted)")
    p.add_argument("--save-nifti", action="store_true", help="Save probability map as NIfTI")
    p.add_argument("--save-npy", action="store_true", help="Save probability map as .npy")
    p.add_argument("--mapping", type=str, default=None, help="Optional aneurysm mapping JSON for distance calculation")
    p.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda")
    return p.parse_args()


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = np.percentile(vol, (1, 99))
    vol = np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return vol.astype(np.float32)


def load_weights(path: str):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def load_mapping(mapping_path: str | None):
    if not mapping_path or not os.path.isfile(mapping_path):
        return []
    try:
        data = json.load(open(mapping_path, "r"))
        coords = []
        for entry in data:
            if "voxel" in entry and len(entry["voxel"]) == 3:
                coords.append(tuple(entry["voxel"]))
        return coords
    except Exception:
        return []


def euclidean(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def save_slice_png(base: np.ndarray, prob: np.ndarray, axis: int, idx: int, out_path: str, cmap: str = "hot", alpha: float = 0.5):
    slice_base = np.take(base, idx, axis=axis)
    slice_prob = np.take(prob, idx, axis=axis)
    plt.figure(figsize=(5,5))
    plt.imshow(slice_base, cmap="gray")
    plt.imshow(slice_prob, cmap=cmap, alpha=alpha)
    plt.axis("off")
    plt.title(f"Axis {axis} slice {idx}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    args = parse_args()
    # Derive UID from volume filename (before _res.nii / _res.nii.gz)
    fname = os.path.basename(args.volume)
    m = re.match(r"(.*?)_res\.nii(\.gz)?$", fname)
    series_uid = m.group(1) if m else os.path.splitext(fname)[0]
    auto_out_dir = os.path.join("src", "app", "results", series_uid)
    out_dir = args.out_dir if args.out_dir else auto_out_dir
    os.makedirs(out_dir, exist_ok=True)

    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available() else
        torch.device(args.device if args.device != "auto" else "cpu")
    )
    print(f"[INFO] Using device: {device}")

    img = nib.load(args.volume)
    vol = img.get_fdata()
    vol_norm = normalize_volume(vol)
    mask = (vol_norm > 0).astype(np.uint8)

    if args.model_type == "simple3dcnn":
        model = AneurysmSimple3DCNN()
    else:
        model = AneurysmTinyUNet3D(base_filters=args.tinyunet_base_filters)

    state = load_weights(args.weights)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    patch_gen = PatchGenerator(patch_size=args.patch_size, stride=args.stride)
    coords = patch_gen.generate_from_mask(mask)
    if not coords:
        raise RuntimeError("No patch coordinates generated. Check stride/patch-size or volume content.")

    engine = ScanInferenceEngine(model, device=device, batch_size=args.batch_size)
    prob_map = engine.infer_scan(vol_norm, coords, patch_gen)

    scan_eval = ScanLevelEvaluator(threshold=args.threshold, aggregation=args.aggregation)
    pred_label, score, pred_coord = scan_eval.predict_scan(prob_map)

    true_coords = load_mapping(args.mapping)
    distance = None
    if true_coords:
        distance = min(euclidean(pred_coord, tc) for tc in true_coords)

    # Save probability map files
    if args.save_nifti:
        prob_img = nib.Nifti1Image(prob_map.astype(np.float32), img.affine)
        nifti_path = os.path.join(out_dir, "prob_map.nii.gz")
        nib.save(prob_img, nifti_path)
        print(f"[INFO] Saved NIfTI prob map -> {nifti_path}")
    if args.save_npy:
        npy_path = os.path.join(out_dir, "prob_map.npy")
        np.save(npy_path, prob_map.astype(np.float32))
        print(f"[INFO] Saved NumPy prob map -> {npy_path}")

    # Mid-slice overlays
    mid_axial = prob_map.shape[2] // 2
    mid_coronal = prob_map.shape[1] // 2
    mid_sagittal = prob_map.shape[0] // 2
    save_slice_png(vol_norm, prob_map, axis=2, idx=mid_axial, out_path=os.path.join(out_dir, "axial_overlay.png"))
    save_slice_png(vol_norm, prob_map, axis=1, idx=mid_coronal, out_path=os.path.join(out_dir, "coronal_overlay.png"))
    save_slice_png(vol_norm, prob_map, axis=0, idx=mid_sagittal, out_path=os.path.join(out_dir, "sagittal_overlay.png"))
    print("[INFO] Saved overlay PNGs (axial/coronal/sagittal).")

    # Threshold mask overlays
    mask_thr = (prob_map >= args.threshold).astype(np.uint8)
    plt.figure(figsize=(5,5))
    plt.imshow(np.take(mask_thr, mid_axial, axis=2), cmap="Reds")
    plt.axis("off")
    plt.title("Axial threshold mask")
    thr_path = os.path.join(out_dir, "axial_threshold_mask.png")
    plt.tight_layout(); plt.savefig(thr_path); plt.close()
    print(f"[INFO] Saved threshold mask -> {thr_path}")

    # Summary JSON
    summary = {
        "volume": args.volume,
        "weights": args.weights,
        "model_type": args.model_type,
        "tinyunet_base_filters": args.tinyunet_base_filters if args.model_type == "tinyunet3d" else None,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "aggregation": args.aggregation,
        "threshold": args.threshold,
        "score": float(score),
        "pred_label": int(pred_label),
        "pred_coord": [int(c) for c in pred_coord],
        "distance_to_nearest_truth": distance,
        "device": str(device),
        "overlay_files": [
            "axial_overlay.png",
            "coronal_overlay.png",
            "sagittal_overlay.png",
            "axial_threshold_mask.png",
        ],
    }
    json_path = os.path.join(out_dir, "visualization_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[INFO] Saved summary JSON -> {json_path}\n[RESULT] label={pred_label} score={score:.4f} coord={pred_coord} distance={distance} out_dir={out_dir}"
    )


if __name__ == "__main__":
    main()
