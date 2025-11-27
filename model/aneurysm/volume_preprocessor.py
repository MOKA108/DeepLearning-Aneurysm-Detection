import numpy as np
import nibabel as nib
from collections import OrderedDict

class VolumePreprocessor:
    """Load, normalize and cache NIfTI volumes for patch extraction.

    A memory-friendly cache is provided:
        - `preload=True`: keep all loaded volumes in RAM (fast, high memory).
        - `preload=False`: maintain a bounded LRU-like temporary cache of size `temp_cache_size`.
    """

    def __init__(self, preload=False, cache_dtype="float32", temp_cache_size=2):
        self.preload = preload
        self.cache_dtype = cache_dtype
        self.cache = {}  # used only if preload True
        self.temp_cache = OrderedDict()  # path -> (vol, mask, coords)
        self.temp_cache_size = max(0, temp_cache_size)

    def load_volume(self, path: str):
        """Load a NIfTI volume, normalize and compute brain mask/coords.

        Returns:
            tuple: (vol, mask, coords) where `vol` is normalized float array,
                   `mask` is uint8 brain mask, and `coords` are precomputed brain voxel indices.
        """
        # Full preload mode: keep everything
        if self.preload:
            if path not in self.cache:
                img = nib.load(path)
                vol = img.get_fdata().astype("float32")
                vol = self.normalize(vol)
                mask = self.compute_brain_mask(vol)
                coords = self.compute_coords(mask)
                if self.cache_dtype == "float16":
                    vol = vol.astype("float16")
                self.cache[path] = (vol, mask, coords)
            return self.cache[path]
        # Temp cache path: reuse a few recent volumes only
        if path in self.temp_cache:
            vol, mask, coords = self.temp_cache[path]
            # move to end (recently used)
            self.temp_cache.move_to_end(path)
            return vol, mask, coords
        img = nib.load(path)
        vol = img.get_fdata().astype("float32")
        vol = self.normalize(vol)
        mask = self.compute_brain_mask(vol)
        coords = self.compute_coords(mask)
        # store in bounded temp cache
        if self.temp_cache_size > 0:
            self.temp_cache[path] = (vol, mask, coords)
            self.temp_cache.move_to_end(path)
            if len(self.temp_cache) > self.temp_cache_size:
                # pop oldest
                self.temp_cache.popitem(last=False)
        return vol, mask, coords

    def normalize(self, vol: np.ndarray) -> np.ndarray:
        """Simple percentile-based normalization into [0,1]."""
        vmin, vmax = np.percentile(vol, (1, 99))
        vol = np.clip((vol - vmin) / (vmax - vmin + 1e-8), 0, 1)
        return vol

    def compute_brain_mask(self, vol):
        """Return a binary mask of brain voxels using a simple threshold (>0)."""
        threshold = 0
        mask = (vol > threshold).astype(np.uint8)
        return mask

    def compute_coords(self, mask):
        # Precompute coordinates of brain voxels once per loaded volume to avoid repeated np.argwhere calls.
        return np.argwhere(mask == 1)
