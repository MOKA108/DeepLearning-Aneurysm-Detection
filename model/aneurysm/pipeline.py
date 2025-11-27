import os
import time
import torch
import csv
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from collections import Counter

from .data_manager import DataManager
from .volume_preprocessor import VolumePreprocessor
from .patch_extractor import PatchExtractor
from .patch_generator import PatchGenerator
from .scan_inference import ScanInferenceEngine
from .scan_level_evaluator import ScanLevelEvaluator
from .dataset import AneurysmPatchDataset
from .model import AneurysmSimple3DCNN, AneurysmTinyUNet3D
from .trainer import AneurysmTrainer
from .evaluator import AneurysmEvaluator
from .visualizer import Visualizer
from .losses import build_loss

class AneurysmPipeline:
    def __init__(self,
                 base_path,
                 model_type: object = "simple",
                 sample_size=None,
                 batch=8,
                 epochs=20,
                 patch_size=32,
                 stride=16,
                 neg=1,
                 num_positive=3,
                 jitter_range=2,
                 preload=True,
                 temp_cache_size=2,
                 amp=False,
                 inference_batch_size=32,
                 num_workers=0,
                 log_gpu=False,
                 patch_dtype="float32",
                 cache_dtype="float32",
                 early_stop_patience=5,
                 monitor_metric="auc",
                 early_stop_mode="max",
                 run_scan_eval_at_end=True,
                 calibrate_scan_threshold=True,
                 scan_aggregation="max",
                 scan_loc_tolerance_vox=None,
                 scan_calibration_objective="youden",
                 loss_type: str = "bce",
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 tinyunet_base_filters: int = 8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using device: {self.device}")
        res = os.path.join(base_path, "data", "preprocessed_data", "resample")
        mapd = os.path.join(base_path, "data", "preprocessed_data", "mapping")
        df = DataManager(res, mapd).build_dataframe()
        if len(df) == 0:
            raise ValueError("No scans found. Check resample/mapping directories under base path.")
        if sample_size is not None:
            if sample_size > len(df):
                print(f"[WARN] Requested sample_size {sample_size} exceeds dataset size {len(df)}; using full dataset.")
            else:
                df = df.sample(n=sample_size, random_state=1234).reset_index(drop=True)
                print(f"Using a sample of size {sample_size} for quick testing.")
        # 70/20/10 split (train/val/test) with stratification fallback
        stratify_col = df["Label"] if df["Label"].value_counts().min() >= 2 else None
        if stratify_col is None:
            print("[WARN] Not enough samples per class for stratified split; using random split.")
        train_df, temp_df = train_test_split(df, test_size=0.30, stratify=stratify_col, random_state=1234)
        val_df, test_df = train_test_split(temp_df, test_size=1/3, stratify=temp_df["Label"] if stratify_col is not None else None, random_state=42)
        print(f"[INFO] Train scans: {len(train_df)} | Val scans: {len(val_df)} | Test scans: {len(test_df)}")
        self.preload = preload
        self.amp = amp
        self.num_workers = num_workers
        self.log_gpu = log_gpu
        self.inference_batch_size = inference_batch_size
        self.stride = stride
        self.early_stop_patience = early_stop_patience
        self.monitor_metric = monitor_metric
        self.early_stop_mode = early_stop_mode
        self.run_scan_eval_at_end = run_scan_eval_at_end
        self.calibrate_scan_threshold = calibrate_scan_threshold
        self.scan_calibration_objective = scan_calibration_objective

        volprep = VolumePreprocessor(preload=self.preload, cache_dtype=cache_dtype, temp_cache_size=temp_cache_size)
        extractor = PatchExtractor(patch_size=patch_size)
        self.patch_generator = PatchGenerator(patch_size=patch_size, stride=stride)

        self.train_set = AneurysmPatchDataset(train_df, volprep, extractor, num_negative=neg, num_positive=num_positive, jitter_range=jitter_range, patch_dtype=patch_dtype)
        self.val_set = AneurysmPatchDataset(val_df, volprep, extractor, num_negative=neg, num_positive=num_positive, jitter_range=jitter_range, patch_dtype=patch_dtype)
        self.test_df = test_df
        print(f"[INFO] Training samples: {len(self.train_set)}, Validation samples: {len(self.val_set)}")

        def count_labels_from_index(dataset):
            c = Counter(lbl for _, lbl, _ in dataset.samples)
            return c.get(1, 0), c.get(0, 0)
        train_pos, train_neg = count_labels_from_index(self.train_set)
        val_pos, val_neg = count_labels_from_index(self.val_set)
        print(f"Training set: {train_pos} positive patches, {train_neg} negative patches")
        print(f"Validation set: {val_pos} positive patches, {val_neg} negative patches")

        self.train_loader = DataLoader(self.train_set, batch_size=batch, shuffle=True, num_workers=self.num_workers, pin_memory=self.device=="cuda")
        self.val_loader = DataLoader(self.val_set, batch_size=batch, num_workers=self.num_workers, pin_memory=self.device=="cuda")
        
        # Instantiate model according to requested type. `model_type` may be:
        #  - a string: 'simple3dcnn' (default) or 'tinyunet3d'
        #  - a callable/class: a PyTorch nn.Module constructor (for further implementations)
        chosen_mt = None
        if isinstance(model_type, str):
            mt = model_type.lower()
            chosen_mt = mt
            if mt in ("simple3dcnn"):
                model_ctor = AneurysmSimple3DCNN
            elif mt in ("tinyunet3d"):
                model_ctor = AneurysmTinyUNet3D
            else:
                raise ValueError(f"Unknown model_type '{model_type}'. Supported: 'simple3dcnn', 'tinyunet3d', or provide a callable class")
        elif callable(model_type):
            model_ctor = model_type
        else:
            raise TypeError("model_type must be a string or callable constructor")
        # Instantiate model, passing base_filters if tiny UNet was requested
        if chosen_mt == "tinyunet3d":
            self.model = model_ctor(base_filters=tinyunet_base_filters).to(self.device)
        else:
            self.model = model_ctor().to(self.device)
        # Record model class for run configuration
        try:
            model_class = f"{self.model.__class__.__module__}.{self.model.__class__.__name__}"
        except Exception:
            model_class = str(type(self.model))

        # Build loss function according to configuration (BCE or Focal)
        self.criterion = build_loss(loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma)

        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.trainer = AneurysmTrainer(self.model, self.optimizer, self.criterion, self.device, amp=self.amp)
        # Evaluator with weighted scoring penalizing false negatives
        self.evaluator = AneurysmEvaluator(
            self.model,
            self.criterion,
            self.device,
            amp=self.amp,
            weights={"TP": 1.0, "TN": 0.1, "FP": 1.0, "FN": 5.0},
            threshold=0.5,
            beta=3.0,
        )
        self.scan_engine = ScanInferenceEngine(self.model, self.device, batch_size=self.inference_batch_size)
        self.scan_eval = ScanLevelEvaluator(threshold=0.5, aggregation=scan_aggregation, loc_tolerance_vox=scan_loc_tolerance_vox)

        self.epochs = epochs
        self.train_losses = []
        self.val_losses = []
        self.val_accs = []
        self.val_sens = []
        self.val_specs = []
        self.val_aucs = []
        self.val_wscores = []
        self.val_fbetas = []
        self.visualizer = Visualizer(self.train_losses, self.val_losses, self.val_accs, self.val_aucs, self.val_sens, self.val_specs, self.val_wscores, self.val_fbetas)
        self.best_metric = -float('inf') if self.early_stop_mode == 'max' else float('inf')
        self.epochs_since_improvement = 0
        self.best_ckpt_path = None

        # Configuration summary to be exported
        self.config = {
            "base_path": base_path,
            "sample_size": sample_size,
            "batch": batch,
            "epochs": epochs,
            "patch_size": patch_size,
            "neg": neg,
            "num_positive": num_positive,
            "jitter_range": jitter_range,
            "stride": stride,
            "inference_batch_size": inference_batch_size,
            "early_stop_patience": early_stop_patience,
            "monitor_metric": monitor_metric,
            "early_stop_mode": early_stop_mode,
            "preload": preload,
            "temp_cache_size": temp_cache_size,
            "amp": amp,
            "num_workers": num_workers,
            "log_gpu": log_gpu,
            "patch_dtype": patch_dtype,
            "cache_dtype": cache_dtype,
            "device": self.device,
            "train_scans": len(train_df),
            "val_scans": len(val_df),
            "train_samples": len(self.train_set),
            "val_samples": len(self.val_set),
            "train_positive_patches": train_pos,
            "train_negative_patches": train_neg,
            "val_positive_patches": val_pos,
            "val_negative_patches": val_neg,
            "scan_aggregation": scan_aggregation,
            "scan_loc_tolerance_vox": scan_loc_tolerance_vox,
            "calibrate_scan_threshold": calibrate_scan_threshold,
            "scan_calibration_objective": scan_calibration_objective,
            "model_class": model_class,
            "calibrated_scan_threshold": None,
            "loss_type": loss_type,
            "focal_alpha": focal_alpha,
            "focal_gamma": focal_gamma,
            "tinyunet_base_filters": tinyunet_base_filters,
        }

    def calibrate_scan_level_threshold(self):
        """Compute scan-level scores on validation scans and calibrate threshold.
        Uses chosen aggregation (mean/max/p95). Requires that self.scan_eval is configured.
        """
        import pandas as pd
        scores = []
        labels = []
        if len(self.val_set) == 0:
            print("[SCAN-CAL] No validation patches available; skipping calibration.")
            return False
        # Build unique list of validation scans from original val dataframe reconstruction
        # We stored val_scans count; need val_df again -> rebuild via underlying DataManager split approach.
        # Simpler: reuse val_set internal reference to dataframe via attribute if present.
        if hasattr(self.val_set, 'df'):
            val_df_ref = self.val_set.df
        else:
            print("[SCAN-CAL] Validation dataframe reference not found; skipping calibration.")
            return False
        print(f"[SCAN-CAL] Computing scan-level scores for {len(val_df_ref)} validation scans...")
        vp = VolumePreprocessor(preload=self.preload)
        for i, row in enumerate(val_df_ref.iterrows()):
            _, r = row
            vol, mask, coords = vp.load_volume(r['Filepath'])
            patch_coords = self.patch_generator.generate_from_mask(mask)
            prob_map = self.scan_engine.infer_scan(vol, patch_coords, self.patch_generator)
            score = self.scan_eval.scan_score(prob_map)
            scores.append(score)
            labels.append(int(r['Label']))
        old_thr = self.scan_eval.threshold
        new_thr = self.scan_eval.fit_threshold(scores, labels, objective=self.scan_calibration_objective)
        print(f"[SCAN-CAL] Threshold calibrated from {old_thr:.4f} to {new_thr:.4f} using objective={self.scan_calibration_objective}.")
        # Save validation scan scores
        cal_df = pd.DataFrame({'score': scores, 'label': labels})
        return cal_df

    def run(self):
        import datetime
        start_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_root = os.path.join(os.getcwd(), "results")
        run_dir = os.path.join(results_root, start_ts)
        os.makedirs(run_dir, exist_ok=True)
        metrics_path = os.path.join(run_dir, "training_metrics.csv")
        config_path = os.path.join(run_dir, "run_config.json")
        # Write config JSON once
        with open(config_path, "w", encoding="utf-8") as cf:
            json.dump(self.config, cf, indent=2)
        if not os.path.exists(metrics_path):
            with open(metrics_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "acc",
                    "sens",
                    "spec",
                    "auc",
                    "weighted_score",
                    "fbeta",
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                ])
        for ep in range(1, self.epochs + 1):
            print(f"[INFO] Starting epoch {ep}/{self.epochs}")
            tr_loss = self.trainer.train_one_epoch(self.train_loader)
            val_loss, acc, sens, spec, auc, conf, wscore, fbeta = self.evaluator.evaluate(self.val_loader)
            self.train_losses.append(tr_loss)
            self.val_losses.append(val_loss)
            self.val_accs.append(acc)
            self.val_sens.append(sens)
            self.val_specs.append(spec)
            self.val_aucs.append(auc)
            self.val_wscores.append(wscore)
            self.val_fbetas.append(fbeta)
            # Save checkpoint
            checkpoint = {
                "epoch": ep,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "metrics": {
                    "acc": acc,
                    "sens": sens,
                    "spec": spec,
                    "auc": auc,
                    "weighted_score": wscore,
                    "fbeta": fbeta,
                    "confusion": conf,
                },
            }
            # Checkpoint filename inside run directory (no extra timestamp, deterministic per epoch)
            ckpt_path = os.path.join(run_dir, f"aneurysm_epoch{ep:03d}.pth")
            torch.save(checkpoint, ckpt_path)
            print(f"[INFO] Saved checkpoint: {ckpt_path}")
            # Append metrics row
            with open(metrics_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    ep,
                    f"{tr_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{acc:.6f}",
                    f"{sens:.6f}",
                    f"{spec:.6f}",
                    f"{auc if auc is not None else ''}",
                    f"{wscore:.6f}",
                    f"{fbeta:.6f}",
                    conf["TP"],
                    conf["TN"],
                    conf["FP"],
                    conf["FN"],
                ])
            # Early stopping monitor
            # NOTE: monitoring uses 'fbeta' by default in current config (was 'auc' previously)
            # current_metric = auc if self.monitor_metric == 'auc' else val_loss
            current_metric = fbeta
            if self.monitor_metric == 'auc':
                current_metric = auc
            elif self.monitor_metric == 'wscore':
                current_metric = wscore
            # current_metric = fbeta if self.monitor_metric == 'fbeta' else auc
            improved = (current_metric > self.best_metric) if self.early_stop_mode == 'max' else (current_metric < self.best_metric)
            if improved:
                self.best_metric = current_metric
                self.epochs_since_improvement = 0
                self.best_ckpt_path = os.path.join(run_dir, "best_model.pth")
                torch.save(self.model.state_dict(), self.best_ckpt_path)
                print(f"[INFO] New best model saved at epoch {ep}")
            else:
                self.epochs_since_improvement += 1
                print(f"[INFO] No improvement for {self.epochs_since_improvement} epoch(s)")
                if self.epochs_since_improvement >= self.early_stop_patience:
                    print(f"[INFO] Early stopping triggered at epoch {ep}")
                    break

            if self.log_gpu and self.device == "cuda":
                allocated = torch.cuda.memory_allocated() / (1024**2)
                reserved = torch.cuda.memory_reserved() / (1024**2)
                gpu_info = f" | gpu_mem alloc={allocated:.1f}MB reserved={reserved:.1f}MB"
            else:
                gpu_info = ""
            print(
                f"Epoch {ep:03d} | train={tr_loss:.4f} | val={val_loss:.4f} | "
                f"acc={acc:.3f} | sens={sens:.3f} | spec={spec:.3f} | auc={auc} | wscore={wscore:.3f} | fbeta={fbeta:.3f}{gpu_info}"
            )
        self.visualizer.plot_learning()

        # Scan-level evaluation
        if self.run_scan_eval_at_end:
            scan_results = []
            if self.best_ckpt_path and os.path.exists(self.best_ckpt_path):
                self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.device))
                print("[INFO] Loaded best model for scan-level evaluation.")
            # Optional calibration step BEFORE evaluating on test set
            if self.calibrate_scan_threshold:
                cal_df = self.calibrate_scan_level_threshold()
                if cal_df is not False:
                    cal_csv = os.path.join(run_dir, "scan_level_calibration_val.csv")
                    cal_df.to_csv(cal_csv, index=False)
                    # Update config with calibrated threshold
                    self.config["calibrated_scan_threshold"] = self.scan_eval.threshold
                    # Persist updated config JSON
                    config_path = os.path.join(run_dir, "run_config.json")
                    try:
                        with open(config_path, "w", encoding="utf-8") as cf:
                            json.dump(self.config, cf, indent=2)
                        print(f"[INFO] Updated run_config.json with calibrated threshold={self.scan_eval.threshold:.4f}")
                    except Exception as e:
                        print(f"[WARN] Failed to update run_config.json: {e}")
                    # Produce histogram (CSV + optional PNG)
                    scores = cal_df['score'].to_numpy()
                    hist_csv = os.path.join(run_dir, "scan_level_calibration_hist.csv")
                    import numpy as np
                    bins = np.linspace(0.0, 1.0, 51)
                    counts, edges = np.histogram(scores, bins=bins)
                    with open(hist_csv, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["bin_left", "bin_right", "count"])
                        for b_left, b_right, c in zip(edges[:-1], edges[1:], counts):
                            w.writerow([f"{b_left:.4f}", f"{b_right:.4f}", c])
                    print(f"[INFO] Saved histogram CSV -> {hist_csv}")
                    # Optional PNG using matplotlib
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(6,4))
                        plt.hist(scores, bins=edges, edgecolor='black')
                        plt.axvline(self.scan_eval.threshold, color='red', linestyle='--', label=f"thr={self.scan_eval.threshold:.3f}")
                        plt.title("Validation Scan-Level Score Distribution")
                        plt.xlabel("Score")
                        plt.ylabel("Count")
                        plt.legend()
                        hist_png = os.path.join(run_dir, "scan_level_calibration_hist.png")
                        plt.tight_layout()
                        plt.savefig(hist_png)
                        plt.close()
                        print(f"[INFO] Saved histogram PNG -> {hist_png}")
                    except Exception as e:
                        print(f"[WARN] Could not create histogram PNG (matplotlib missing?): {e}")
                    print(f"[INFO] Saved calibration validation scores -> {cal_csv}")
            for i, row in enumerate(self.test_df.iterrows()):
                r_index, r = row
                print(f"[SCAN-EVAL] ({i+1}/{len(self.test_df)}) {os.path.basename(r['Filepath'])}")
                vol, mask, coords = VolumePreprocessor(preload=self.preload).load_volume(r['Filepath'])
                patch_coords = self.patch_generator.generate_from_mask(mask)
                prob_map = self.scan_engine.infer_scan(vol, patch_coords, self.patch_generator)
                eval_res = self.scan_eval.evaluate_scan(prob_map, r['AneurysmCoords'])
                eval_res['uid'] = r['SeriesInstanceUID']
                eval_res['label'] = r['Label']
                scan_results.append(eval_res)
            import pandas as pd
            scan_df = pd.DataFrame(scan_results)
            agg = ScanLevelEvaluator.aggregate(scan_df)
            print(f"[SCAN-AGG] acc={agg['acc']:.3f} sens={agg['sens']:.3f} spec={agg['spec']:.3f} auc={agg['auc']}")
            
            # Persist per-scan detailed results
            scan_csv = os.path.join(run_dir, "scan_level_results.csv")
            scan_df.to_csv(scan_csv, index=False)
            print(f"[INFO] Saved scan-level results -> {scan_csv}")

            # Persist aggregated scan-level metrics (single-row CSV)
            scan_metrics_csv = os.path.join(run_dir, "scan_level_metrics.csv")
            try:
                with open(scan_metrics_csv, "w", newline="") as mf:
                    mw = csv.writer(mf)
                    mw.writerow(["acc", "sens", "spec", "auc"])
                    mw.writerow([
                        f"{agg.get('acc', 0.0):.6f}",
                        f"{agg.get('sens', 0.0):.6f}",
                        f"{agg.get('spec', 0.0):.6f}",
                        f"{agg.get('auc', '') if agg.get('auc', None) is not None else ''}"
                    ])
                print(f"[INFO] Saved scan-level metrics -> {scan_metrics_csv}")
            except Exception as e:
                print(f"[WARN] Could not save scan-level metrics CSV: {e}")
        return True
