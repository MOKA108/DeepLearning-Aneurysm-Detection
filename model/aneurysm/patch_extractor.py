import numpy as np
import random


class PatchExtractor:
    """Utility to extract cubic patches from 3D volumes.

    Provides safe extraction with zero-padding near borders, a random negative coordinate
    sampler, and a simple informativeness test to avoid empty negatives.
    """

    def __init__(self, patch_size=32, seed=123):
        self.patch_size = patch_size
        self.rng = random.Random(seed)

    def extract_patch(self, vol, x, y, z):
        """Extract a cubic patch centered (approximately) at (x,y,z).

        Pads with zeros if the patch would extend beyond volume boundaries.
        """
        p = self.patch_size
        r = p // 2
        x, y, z = int(x), int(y), int(z)
        patch = vol[
            max(0, x - r) : x + r,
            max(0, y - r) : y + r,
            max(0, z - r) : z + r,
        ]
        patch = np.pad(
            patch,
            (
                (0, max(0, p - patch.shape[0])),
                (0, max(0, p - patch.shape[1])),
                (0, max(0, p - patch.shape[2])),
            ),
            mode="constant",
        )
        return patch[:p, :p, :p]

    def random_negative_coord(self, brain_mask, precomputed_coords=None):
        """Return a random coordinate inside the brain mask.

        If `precomputed_coords` is provided, use it to avoid repeated calls to `np.argwhere`.
        """
        if precomputed_coords is None:
            precomputed_coords = np.argwhere(brain_mask == 1)
        idx = self.rng.randint(0, len(precomputed_coords) - 1)
        x, y, z = precomputed_coords[idx]
        return int(x), int(y), int(z)

    def is_informative(self, patch, threshold=0.75):
        """Return True if a patch contains sufficient non-zero voxels.

        This simple heuristic helps avoid empty background negatives.
        """
        return np.mean(patch > 0) > threshold
