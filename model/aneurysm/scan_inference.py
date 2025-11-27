import numpy as np
import torch

class ScanInferenceEngine:
    """Run batched inference over patches and aggregate results into a probability map.

    The engine converts a list of patch coordinates into patch arrays, runs the model in batches,
    and writes the maximum probability observed per voxel into a probability map.
    """
    def __init__(self, model, device, batch_size=32):
        self.model = model
        self.device = device
        self.batch_size = int(batch_size)

    def infer_scan(self, vol: np.ndarray, coords, patch_generator):
        p = patch_generator.patch_size
        patches = patch_generator.generate_patches_from_vol(vol, coords)
        prob_map = np.zeros_like(vol, dtype=np.float32)
        # count_map = np.zeros_like(vol, dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch = patches[i:i+self.batch_size]
                batch_tensor = torch.from_numpy(np.stack(batch, axis=0)).to(self.device).unsqueeze(1).float()
                logits = self.model(batch_tensor).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                for j, (x, y, z) in enumerate(coords[i:i+self.batch_size]):
                    prob = float(probs[j])
                    patch_slice = (slice(x, x+p), slice(y, y+p), slice(z, z+p))
                    prob_map[patch_slice] = np.maximum(prob_map[patch_slice], prob)
                    # count_map[patch_slice] += 1.0

        # Note: We intentionally aggregate using max-probability per voxel to preserve localized spikes.
        # An alternative would be an average across overlapping patches (commented out below).
        return prob_map
