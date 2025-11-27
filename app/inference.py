"""Standalone inference script for aneurysm detection using a trained model.

Usage (PowerShell example):
  python ./src/app/inference.py `
    --volume "C:/path/to/scan_res.nii.gz" `
    --model-type tinyunet3d `
    --tinyunet-base-filters 16 `
    --patch-size 32 `
    --stride 16 `
    --threshold 0.5 `
    --output ./prediction.json

The script:
  1. Loads the trained weights from best_model.pth (in same folder by default)
  2. Instantiates the requested model architecture
  3. Normalizes the volume similarly to training (percentile 1-99 scaling)
  4. Generates a patch grid and runs batched inference
  5. Aggregates voxel probabilities (max|mean|p95) into a scan score
  6. Applies threshold to get a binary label
  7. Saves JSON output with score, label, predicted coordinate

Optionally pass --mapping to supply a JSON aneurysm coordinate mapping file; distance to predicted coord will be reported.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import torch
import nibabel as nib

# Ensure we can import the aneurysm package modules
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from aneurysm.model import AneurysmSimple3DCNN, AneurysmTinyUNet3D
from aneurysm.patch_generator import PatchGenerator
from aneurysm.scan_inference import ScanInferenceEngine
from aneurysm.scan_level_evaluator import ScanLevelEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="Run scan-level aneurysm inference")
    p.add_argument("--volume", required=True, type=str, help="Path to resampled NIfTI volume (.nii/.nii.gz)")
    p.add_argument("--model-type", type=str, default="tinyunet3d", choices=["simple3dcnn", "tinyunet3d"], help="Model architecture")
    p.add_argument("--tinyunet-base-filters", type=int, default=8, help="Base filters for TinyUNet (training used 8 if not specified in config)")
    p.add_argument("--patch-size", type=int, default=32, help="Patch edge length")
    p.add_argument("--stride", type=int, default=10, help="Stride for patch grid (training default 10)")
    p.add_argument("--aggregation", type=str, default="max", choices=["max", "mean", "p95"], help="Scan-level aggregation strategy (training used max)")
    p.add_argument("--threshold", type=float, default=0.85, help="Decision threshold on aggregated score (calibrated training threshold 0.85)")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for patch inference (training inference batch size)")
    p.add_argument("--device", type=str, default="auto", help="'auto' | 'cpu' | 'cuda' (if available)")
    p.add_argument("--output", type=str, default="prediction.json", help="Output JSON file path")
    p.add_argument("--weights", type=str, default="best_model.pth", help="Path to model weights (.pth)")
    p.add_argument("--mapping", type=str, default=None, help="Optional JSON mapping file with aneurysm coordinates to compute distance")
    return p.parse_args()


def load_weights(weights_path: str) -> dict | torch.Tensor:
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vmin, vmax = np.percentile(vol, (1, 99))
    vol = np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)
    return vol.astype(np.float32)


def brain_mask(vol: np.ndarray) -> np.ndarray:
    return (vol > 0).astype(np.uint8)


def load_mapping(mapping_path: str | None) -> list[tuple[float, float, float]]:
    if not mapping_path:
        return []
    if not os.path.isfile(mapping_path):
        print(f"[WARN] Mapping file not found: {mapping_path}")
        return []
    try:
        data = json.load(open(mapping_path, "r"))
        coords = []
        for entry in data:
            if "voxel" in entry and len(entry["voxel"]) == 3:
                coords.append(tuple(entry["voxel"]))
        return coords
    except Exception as e:
        print(f"[WARN] Failed to parse mapping JSON: {e}")
        return []


def euclidean(a, b) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    return float(np.sqrt(np.sum((a - b) ** 2)))


def main():
    args = parse_args()

    device = (
        torch.device("cuda") if args.device == "auto" and torch.cuda.is_available() else
        torch.device(args.device if args.device != "auto" else "cpu")
    )
    print(f"[INFO] Using device: {device}")

    # Load volume
    img = nib.load(args.volume)
    vol = img.get_fdata()
    vol = normalize_volume(vol)
    mask = brain_mask(vol)

    # Instantiate model
    if args.model_type == "simple3dcnn":
        model = AneurysmSimple3DCNN()
    else:
        model = AneurysmTinyUNet3D(base_filters=args.tinyunet_base_filters)

    state = load_weights(args.weights)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()

    patch_gen = PatchGenerator(patch_size=args.patch_size, stride=args.stride)
    coords = patch_gen.generate_from_mask(mask)
    if len(coords) == 0:
        raise RuntimeError("No patch coordinates generated; check mask or patch/stride settings.")

    engine = ScanInferenceEngine(model, device=device, batch_size=args.batch_size)
    prob_map = engine.infer_scan(vol, coords, patch_gen)

    scan_eval = ScanLevelEvaluator(threshold=args.threshold, aggregation=args.aggregation)
    pred_label, score, pred_coord = scan_eval.predict_scan(prob_map)

    true_coords = load_mapping(args.mapping)
    best_distance = None
    if true_coords:
        best_distance = min(euclidean(pred_coord, tc) for tc in true_coords)

    output = {
        "volume": args.volume,
        "weights": args.weights,
        "model_type": args.model_type,
        "tinyunet_base_filters": args.tinyunet_base_filters if args.model_type == "tinyunet3d" else None,
        "patch_size": args.patch_size,
        "stride": args.stride,
        "aggregation": args.aggregation,
        "threshold": args.threshold,
        "score": score,
        "pred_label": int(pred_label),
        "pred_coord": [int(c) for c in pred_coord],
        "mapping_used": args.mapping,
        "distance_to_nearest_truth": best_distance,
        "device": str(device),
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"[INFO] Saved prediction JSON -> {args.output}")
    print(f"[RESULT] label={pred_label} score={score:.4f} coord={pred_coord} distance={best_distance}")


if __name__ == "__main__":
    main()
