import numpy as np


class PatchGenerator:
    """Produce grid coordinates for patch extraction and build patch arrays from a volume.

    By default produces start indices such that patches fully fit within the volume; stride
    controls overlap (stride < patch_size => overlapping patches).
    """

    def __init__(self, patch_size=32, stride=16):
        self.patch_size = int(patch_size)
        self.stride = int(stride)

    def generate_from_mask(self, mask: np.ndarray):
        """Generate top-left-start coordinates for cubic patches that fit inside the mask volume.

        Args:
            mask: 3D mask or volume shape (D,H,W).

        Returns:
            List of (x,y,z) start indices.
        """
        D, H, W = mask.shape
        p = self.patch_size
        coords = []
        # iterate start indices so patch fits entirely: [0, D-p] step stride
        for x in range(0, max(1, D - p + 1), self.stride):
            for y in range(0, max(1, H - p + 1), self.stride):
                for z in range(0, max(1, W - p + 1), self.stride):
                    coords.append((x, y, z))
        return coords

    def generate_patches_from_vol(self, vol: np.ndarray, coords):
        """Given a set of start coordinates, extract cubic patches from `vol`.

        Pads partial patches at borders with zeros to always return shape `(p,p,p)`.
        """
        p = self.patch_size
        patches = []
        for (x, y, z) in coords:
            patch = vol[x : x + p, y : y + p, z : z + p]
            if patch.shape != (p, p, p):
                pad_x = max(0, p - patch.shape[0])
                pad_y = max(0, p - patch.shape[1])
                pad_z = max(0, p - patch.shape[2])
                patch = np.pad(
                    patch,
                    ((0, pad_x), (0, pad_y), (0, pad_z)),
                    mode="constant",
                    constant_values=0.0,
                )
                patch = patch[:p, :p, :p]
            patches.append(patch)
        return patches
