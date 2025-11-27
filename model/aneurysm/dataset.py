import random
import numpy as np
import torch
from torch.utils.data import Dataset


class AneurysmPatchDataset(Dataset):
    """PyTorch Dataset that yields 3D patches for aneurysm classification.

    Behavior:
        - Positive samples: generated around annotated aneurysm coordinates (with optional jitter).
        - Negative samples: random brain coordinates sampled from precomputed brain mask coords.

    Args:
        df: DataFrame produced by `DataManager` containing filepaths and labels.
        preproc: `VolumePreprocessor` instance used to load/normalize volumes.
        extractor: `PatchExtractor` used to extract cubic patches.
        num_negative: Number of negative patches to sample per scan.
        num_positive: Number of positive patches to sample per annotated coordinate.
        jitter_range: Max voxel jitter applied to positive center coordinates.
        patch_dtype: Desired patch dtype (e.g., "float32" or "float16").
    """

    def __init__(
        self,
        df,
        preproc,
        extractor,
        num_negative=1,
        num_positive=3,
        jitter_range=2,
        patch_dtype="float32",
    ):
        self.df = df
        self.preproc = preproc
        self.extractor = extractor
        self.num_negative = num_negative
        self.num_pos_patches = num_positive
        self.jitter_range = jitter_range
        self.patch_dtype = patch_dtype
        self.samples = []
        for _, row in self.df.iterrows():
            filepath = row["Filepath"]
            label = row["Label"]
            coords = row["AneurysmCoords"]

            if label == 1:
                for c in coords:
                    for _ in range(self.num_pos_patches):
                        self.samples.append((filepath, 1, c))

            for _ in range(self.num_negative):
                self.samples.append((filepath, 0, None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load volume, extract a patch (positive or negative), and return tensor + label.

        Returns:
            tuple: (patch_tensor, label_tensor)
        """
        filepath, label, coord = self.samples[idx]
        vol, brain_mask, coords = self.preproc.load_volume(filepath)
        if label == 1:
            x, y, z = coord
            if self.jitter_range > 0:
                x += random.randint(-self.jitter_range, self.jitter_range)
                y += random.randint(-self.jitter_range, self.jitter_range)
                z += random.randint(-self.jitter_range, self.jitter_range)
                x = np.clip(x, 0, vol.shape[0] - 1)
                y = np.clip(y, 0, vol.shape[1] - 1)
                z = np.clip(z, 0, vol.shape[2] - 1)
            patch = self.extractor.extract_patch(vol, x, y, z)
        else:
            # Sample multiple negatives quickly using precomputed coords without recomputing argwhere each time.
            for _ in range(20):
                x, y, z = self.extractor.random_negative_coord(brain_mask, coords)
                patch = self.extractor.extract_patch(vol, x, y, z)
                if self.extractor.is_informative(patch):
                    break
        # Convert patch to chosen dtype to reduce memory if float16
        tensor = torch.from_numpy(patch)
        if self.patch_dtype == "float16":
            tensor = tensor.to(torch.float16)
        else:
            tensor = tensor.to(torch.float32)
        patch = tensor.unsqueeze(0)
        return patch, torch.tensor(label, dtype=torch.float32)