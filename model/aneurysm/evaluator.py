import torch
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, fbeta_score


class AneurysmEvaluator:
    """Evaluation helper for patch-level classification.

    Computes loss, accuracy, sensitivity, specificity, AUC (if applicable),
    a weighted score, and F-beta metric. Designed to work with a PyTorch DataLoader.
    """

    def __init__(self, model, criterion, device, amp=False, weights=None, threshold=0.5, beta=3.0):
        self.model = model
        self.criterion = criterion
        self.device = device
        self.amp = amp and device == "cuda"
        self.weights = weights or {"TP": 1.0, "TN": 0.1, "FP": 1.0, "FN": 5.0}
        self.threshold = threshold
        self.beta = beta

    def evaluate(self, loader):
        """Evaluate model on a DataLoader.

        Args:
            loader: PyTorch DataLoader yielding (X, y) batches.

        Returns:
            tuple: (avg_loss, accuracy, sensitivity, specificity, auc_or_none,
                   confusion_dict, weighted_score, fbeta)
        """
        self.model.eval()
        losses = []
        preds = []
        labels = []
        scores = []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device)
                target_dtype = next(self.model.parameters()).dtype
                if X.dtype != target_dtype:
                    X = X.to(target_dtype)
                if self.amp:
                    with torch.cuda.amp.autocast():
                        logits = self.model(X).squeeze(1)
                        loss = self.criterion(logits, y)
                else:
                    logits = self.model(X).squeeze(1)
                    loss = self.criterion(logits, y)
                losses.append(loss.item())
                prob = torch.sigmoid(logits)
                batch_preds = (prob >= self.threshold).cpu().numpy()
                preds.extend(batch_preds)
                labels.extend(y.cpu().numpy())
                scores.extend(prob.cpu().numpy())

        if not labels:
            return 0.0, 0.0, 0.0, 0.0, None, {"TP": 0, "TN": 0, "FP": 0, "FN": 0}, 0.0, 0.0

        acc = accuracy_score(labels, preds)
        sens = recall_score(labels, preds) if any(labels) else 0.0
        spec = recall_score(labels, preds, pos_label=0) if any(l == 0 for l in labels) else 0.0
        try:
            auc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else None
        except Exception:
            auc = None

        TP = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        TN = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
        FP = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        FN = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)

        w = self.weights
        weighted_raw = (w["TP"] * TP + w["TN"] * TN) - (w["FP"] * FP + w["FN"] * FN)
        denom = (w["TP"] * TP + w["TN"] * TN + w["FP"] * FP + w["FN"] * FN) or 1.0
        weighted_score = weighted_raw / denom

        try:
            fbeta = fbeta_score(labels, preds, beta=self.beta) if len(set(labels)) > 1 else 0.0
        except Exception:
            fbeta = 0.0

        return (
            sum(losses) / len(losses),
            acc,
            sens,
            spec,
            auc,
            {"TP": TP, "TN": TN, "FP": FP, "FN": FN},
            weighted_score,
            fbeta,
        )
