import matplotlib.pyplot as plt
import numpy as np


class Visualizer:
    """Plotting helpers for training diagnostics and probability maps."""

    def __init__(
        self,
        train_losses=None,
        val_losses=None,
        val_accs=None,
        val_aucs=None,
        val_sens=None,
        val_specs=None,
        val_wscore=None,
        val_fbeta=None,
    ):
        self.train_losses = train_losses if train_losses is not None else []
        self.val_losses = val_losses if val_losses is not None else []
        self.val_accs = val_accs if val_accs is not None else []
        self.val_aucs = val_aucs if val_aucs is not None else []
        self.val_sens = val_sens if val_sens is not None else []
        self.val_specs = val_specs if val_specs is not None else []
        self.val_wscore = val_wscore if val_wscore is not None else []
        self.val_fbeta = val_fbeta if val_fbeta is not None else []

    def plot_learning(self):
        """Plot loss curves and validation metrics across epochs."""
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'o-', label='Train Loss')
        plt.plot(epochs, self.val_losses, 'o-', label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        if self.val_accs:
            plt.plot(epochs, self.val_accs, 'o-', label='Val Acc')
        if self.val_aucs:
            plt.plot(epochs, self.val_aucs, 'o-', label='Val AUC')
        if self.val_sens:
            plt.plot(epochs, self.val_sens, 'o-', label='Val Sens')
        if self.val_specs:
            plt.plot(epochs, self.val_specs, 'o-', label='Val Spec')
        if self.val_wscore:
            plt.plot(epochs, self.val_wscore, 'o-', label='Weighted Score')
        if self.val_fbeta:
            plt.plot(epochs, self.val_fbeta, 'o-', label='F-beta')
        plt.xlabel('Epoch')
        plt.ylabel('Metric')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Backward compatible wrapper
    def plot(self):
        self.plot_learning()

    def plot_prob_slice(self, prob_map, axis=0, idx=None):
        """Show a single slice of a probability map as a heatmap."""
        if idx is None:
            idx = prob_map.shape[axis] // 2
        slice_img = np.take(prob_map, idx, axis=axis)
        plt.figure(figsize=(6, 5))
        plt.title(f"Probability Heatmap axis={axis} slice={idx}")
        plt.imshow(slice_img, cmap='hot')
        plt.colorbar()
        plt.show()
