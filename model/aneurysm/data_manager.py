import os
import json
import pandas as pd


class DataManager:
    """Manage dataset file listings and mapping metadata.

    This helper scans a resampled NIfTI folder and corresponding mapping JSONs to
    build a Pandas DataFrame with per-series information used by model training.
    """

    def __init__(self, resample_dir: str, mapping_dir: str):
        self.RESAMPLE_DIR = resample_dir
        self.MAPPING_DIR = mapping_dir
        self.data = None

    def build_dataframe(self):
        """Scan `RESAMPLE_DIR` and build a DataFrame of series metadata.

        Returns:
            pandas.DataFrame: Columns include `SeriesInstanceUID`, `Filepath`, `MappingPath`,
            `Label` (0/1), and `AneurysmCoords` (list of coordinate tuples).
        """
        rows = []
        for f in sorted(os.listdir(self.RESAMPLE_DIR)):
            if not (f.endswith(".nii.gz") or f.endswith(".nii")):
                continue
            if f.endswith("_res.nii.gz"):
                uid = f[:-len("_res.nii.gz")]
            elif f.endswith("_res.nii"):
                uid = f[:-len("_res.nii")]
            else:
                # Fallback: strip extension generically
                uid = f.rsplit('.', 1)[0]
            nii_path = os.path.join(self.RESAMPLE_DIR, f)
            json_path = os.path.join(self.MAPPING_DIR, f"{uid}_mapping.json")
            label = 0
            coords = []
            if os.path.exists(json_path):
                try:
                    data = json.load(open(json_path))
                    for d in data:
                        if "voxel" in d:
                            coords.append(tuple(d["voxel"]))
                    if len(coords) > 0:
                        label = 1
                except Exception:
                    pass
            rows.append({
                "SeriesInstanceUID": uid,
                "Filepath": nii_path,
                "MappingPath": json_path if os.path.exists(json_path) else None,
                "Label": label,
                "AneurysmCoords": coords,
            })
        self.data = pd.DataFrame(rows)
        return self.data

    def summary(self, head=5):
        if self.data is None:
            print("No data loaded yet. Run build_dataframe() first.")
            return
        print("Dataset summary:")
        print(f"Total series: {len(self.data)}")
        print(self.data["Label"].value_counts())
        print("\nSample rows:")
        print(self.data.head(head))

    def get_dataframe(self):
        if self.data is None:
            raise ValueError("Data not loaded yet — call build_dataframe() first.")
        return self.data
