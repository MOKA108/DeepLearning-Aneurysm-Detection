import argparse
import os
from aneurysm.pipeline import AneurysmPipeline

DEFAULT_BASE = "/mnt/vdb/deep_learning"

def parse_args():
    p = argparse.ArgumentParser(description="Run aneurysm patch classification pipeline")

    # === Paths & Data Root ===
    p.add_argument("--base-path", type=str, default=DEFAULT_BASE, help="Base path containing data directory")

    # === Dataset Sampling / Splits ===
    p.add_argument("--sample-size", type=int, default=None, help="Optional subset of scans for quick tests")

    # === Patch Extraction Parameters ===
    p.add_argument("--patch-size", type=int, default=32, help="3D patch edge length")
    p.add_argument("--neg", type=int, default=1, help="Negative patch placeholders per scan")
    p.add_argument("--num-positive", type=int, default=5, help="Positive patches multiplier per aneurysm")
    p.add_argument("--jitter-range", type=int, default=2, help="Random voxel jitter around positive coords")
    p.add_argument("--patch-dtype", type=str, default="float32", choices=["float32", "float16"], help="Patch tensor dtype (float16 reduces RAM/GPU memory)")

    # === Volume Loading / Caching ===
    p.add_argument("--no-preload", action="store_true", help="Disable volume preloading to save RAM")
    p.add_argument("--temp-cache-size", type=int, default=2, help="Recent volumes kept in RAM if not preloading")
    p.add_argument("--cache-dtype", type=str, default="float32", choices=["float32", "float16"], help="Cached volume dtype when preloading")

    # === Model Architecture Selection ===
    p.add_argument("--model-type", type=str, default="simple3dcnn", choices=["simple3dcnn", "tinyunet3d"], help="Model type to use")
    p.add_argument("--tinyunet-base-filters", type=int, default=8, help="Base filters for TinyUNet (ignored unless tinyunet3d)")

    # === Loss Function ===
    p.add_argument("--loss-type", type=str, default="bce", choices=["bce", "focal"], help="Loss: bce or focal")
    p.add_argument("--focal-alpha", type=float, default=0.25, help="Focal alpha (positive class weight)")
    p.add_argument("--focal-gamma", type=float, default=2.0, help="Focal gamma (hard example focus)")

    # === Training Loop Hyperparameters ===
    p.add_argument("--batch", type=int, default=8, help="Batch size for training")
    p.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision on CUDA")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")

    # === Early Stopping / Monitoring ===
    p.add_argument("--early-stop-patience", type=int, default=5, help="Epochs without improvement before stop")
    p.add_argument("--monitor-metric", type=str, default="auc", choices=["auc", "fbeta", "wscore"], help="Metric to monitor for early stopping")
    p.add_argument("--early-stop-mode", type=str, default="max", choices=["max", "min"], help="Improvement direction for monitored metric")

    # === Scan-Level Inference / Evaluation ===
    p.add_argument("--stride", type=int, default=16, help="Stride for scan-level patch grid generation")
    p.add_argument("--inference-batch-size", type=int, default=32, help="Batch size for scan-level inference")
    p.add_argument("--no-scan-eval", action="store_true", help="Disable final scan-level evaluation phase")
    p.add_argument("--scan-aggregation", type=str, default="max", choices=["max", "mean", "p95"], help="Aggregation strategy for scan score")
    p.add_argument("--scan-calibration-objective", type=str, default="youden", choices=["youden", "fbeta"], help="Threshold calibration objective")
    p.add_argument("--no-calibrate-scan-threshold", action="store_true", help="Disable validation threshold calibration")
    p.add_argument("--scan-loc-tolerance-vox", type=int, default=None, help="Voxel tolerance for localization correctness")

    # === Logging / Diagnostics ===
    p.add_argument("--log-gpu", action="store_true", help="Log GPU memory each epoch (CUDA only)")

    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.isdir(args.base_path):
        raise SystemExit(f"Base path {args.base_path} not found")
    pipeline = AneurysmPipeline(
        base_path=args.base_path,
        model_type=args.model_type,
        sample_size=args.sample_size,
        batch=args.batch,
        epochs=args.epochs,
        patch_size=args.patch_size,
        stride=args.stride,
        neg=args.neg,
        num_positive=args.num_positive,
        jitter_range=args.jitter_range,
        preload=not args.no_preload,
        temp_cache_size=args.temp_cache_size,
        amp=args.amp,
        inference_batch_size=args.inference_batch_size,
        num_workers=args.num_workers,
        log_gpu=args.log_gpu,
        patch_dtype=args.patch_dtype,
        cache_dtype=args.cache_dtype,
        early_stop_patience=args.early_stop_patience,
        monitor_metric=args.monitor_metric,
        early_stop_mode=args.early_stop_mode,
        run_scan_eval_at_end=not args.no_scan_eval,
        calibrate_scan_threshold=not args.no_calibrate_scan_threshold,
        scan_aggregation=args.scan_aggregation,
        scan_calibration_objective=args.scan_calibration_objective,
        scan_loc_tolerance_vox=args.scan_loc_tolerance_vox,
        loss_type=args.loss_type,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        tinyunet_base_filters=args.tinyunet_base_filters,
    )
    pipeline.run()

if __name__ == "__main__":
    main()
