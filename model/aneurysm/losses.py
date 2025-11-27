import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Binary focal loss for addressing class imbalance.

    Args:
        gamma (float): Focusing parameter (typically 2). Higher values focus more on hard examples.
        alpha (float): Balancing factor for positive class (0..1). If None, no class weighting applied.

    Forward expects raw logits and target tensor of same shape with 0/1 labels.
    Returns mean focal loss over batch.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits (no reduction to get per-sample values)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # Probabilities for the positive class
        probs = torch.sigmoid(logits)
        # pt = p if target=1 else (1-p)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        # Focal scaling term (1-pt)^gamma
        focal_term = (1.0 - pt) ** self.gamma
        loss = focal_term * bce
        if self.alpha is not None:
            alpha_factor = torch.where(targets == 1, torch.full_like(targets, self.alpha), torch.full_like(targets, 1 - self.alpha))
            loss = alpha_factor * loss
        return loss.mean()


def build_loss(loss_type: str, focal_alpha: float = 0.25, focal_gamma: float = 2.0):
    """Factory to build a loss instance by name.

    Args:
        loss_type: 'bce' or 'focal'
        focal_alpha: alpha for focal (ignored if not focal)
        focal_gamma: gamma for focal (ignored if not focal)
    """
    lt = (loss_type or "bce").lower()
    if lt == "bce":
        return nn.BCEWithLogitsLoss()
    if lt == "focal":
        return BinaryFocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    raise ValueError(f"Unknown loss_type '{loss_type}'. Expected 'bce' or 'focal'.")
