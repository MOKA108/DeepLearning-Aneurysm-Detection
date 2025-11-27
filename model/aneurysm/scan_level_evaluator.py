import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

class ScanLevelEvaluator:
    """
    Scan-level evaluator with flexible aggregation and optional threshold calibration.

    Aggregation strategies over a voxel probability map (prob_map):
      - 'max': maximum voxel probability (original behavior)
      - 'mean': mean of all voxel probabilities (robust to isolated spikes)
      - 'p95': 95th percentile (compromise between max and mean)

    Threshold calibration can be performed post-training using validation scores via fit_threshold.
    Localization correctness can optionally enforce a distance tolerance (in voxels) for true positive classification.
    """
    def __init__(self, threshold=0.5, aggregation="max", loc_tolerance_vox=None):
        assert aggregation in ("max", "mean", "p95"), "aggregation must be one of: max, mean, p95"
        self.threshold = float(threshold)
        self.aggregation = aggregation
        self.loc_tolerance_vox = loc_tolerance_vox  # None means ignore distance in correctness flag
        self._calibrated = False

    # -------- Core helpers --------
    def scan_score(self, prob_map):
        if self.aggregation == "max":
            return float(np.max(prob_map))
        if self.aggregation == "mean":
            return float(prob_map.mean())
        if self.aggregation == "p95":
            return float(np.percentile(prob_map, 95))
        raise ValueError("Invalid aggregation")

    def predict_scan(self, prob_map):
        score = self.scan_score(prob_map)
        pred_label = int(score >= self.threshold)
        pred_coord = np.unravel_index(int(np.argmax(prob_map)), prob_map.shape)
        return pred_label, score, pred_coord

    @staticmethod
    def distance(c1, c2):
        return float(np.sqrt(np.sum((np.array(c1) - np.array(c2))**2)))

    # -------- Evaluation per scan --------
    def evaluate_scan(self, prob_map, true_coords):
        pred_label, score, pred_coord = self.predict_scan(prob_map)
        # No ground-truth aneurysm coords -> correct if model predicts absence
        if len(true_coords) == 0:
            return {
                "pred": pred_label,
                "max_prob": score,  # keep key name for backward compatibility
                "pred_coord": pred_coord,
                "distance_mm": None,
                "correct": pred_label == 0
            }
        # Compute best distance to any true coordinate
        best_dist = min(self.distance(pred_coord, tc) for tc in true_coords)
        if self.loc_tolerance_vox is not None:
            loc_ok = (pred_label == 1) and (best_dist <= self.loc_tolerance_vox)
        else:
            loc_ok = (pred_label == 1)
        return {
            "pred": pred_label,
            "max_prob": score,
            "pred_coord": pred_coord,
            "distance_mm": best_dist,
            "correct": loc_ok
        }

    # -------- Threshold calibration --------
    def fit_threshold(self, scores, labels, objective="youden", beta=3.0, grid_size=201):
        """
        Calibrate decision threshold using validation scores.
        Parameters:
          scores   : iterable of float scan-level scores (after aggregation)
          labels   : iterable of int {0,1}
          objective: 'youden' (maximize sens+spec-1) or 'fbeta'
          beta     : beta for F-beta when objective='fbeta'
          grid_size: number of thresholds in [0,1] grid search
        Returns: best threshold found (float)
        """
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        assert scores.shape[0] == labels.shape[0] and scores.ndim == 1
        if len(np.unique(labels)) < 2:
            # Cannot calibrate with single-class set
            return self.threshold
        best_thr = self.threshold
        best_val = -1.0
        for t in np.linspace(0.0, 1.0, grid_size):
            pred = (scores >= t).astype(int)
            tp = np.sum((pred == 1) & (labels == 1))
            fp = np.sum((pred == 1) & (labels == 0))
            fn = np.sum((pred == 0) & (labels == 1))
            tn = np.sum((pred == 0) & (labels == 0))
            sens = tp / (tp + fn + 1e-9)
            spec = tn / (tn + fp + 1e-9)
            if objective == "youden":
                val = sens + spec - 1.0
            elif objective == "fbeta":
                val = (1 + beta**2) * tp / ((1 + beta**2) * tp + beta**2 * fn + fp + 1e-9)
            else:
                raise ValueError("objective must be 'youden' or 'fbeta'")
            if val > best_val:
                best_val = val
                best_thr = t
        self.threshold = float(best_thr)
        self._calibrated = True
        return self.threshold

    # -------- Aggregation over result dataframe --------
    @staticmethod
    def aggregate(results_df):
        y_true = results_df["label"].tolist()
        y_pred = results_df["pred"].tolist()
        y_score = results_df["max_prob"].tolist()  # using stored scan-level score
        acc = accuracy_score(y_true, y_pred)
        sens = recall_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
        spec = recall_score(y_true, y_pred, pos_label=0) if len(set(y_true)) > 1 else 0.0
        try:
            auc = roc_auc_score(y_true, y_score) if len(set(y_true)) > 1 else None
        except Exception:
            auc = None
        return {"acc": acc, "sens": sens, "spec": spec, "auc": auc}
