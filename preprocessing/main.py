"""Batch preprocessing driver orchestrating the full MR pipeline.

Sequence (high-level):
    1. DICOM → NIfTI conversion.
    2. Skull stripping (SynthStrip) for brain-only extraction.
    3. N4 bias field correction to mitigate smooth intensity shading.
    4. Intensity normalization (e.g. min-max / z-score).
    5. Resampling to isotropic voxel spacing (e.g. 1 mm³).
    6. Mapping 2D DICOM annotation points into 3D voxel coordinates (original → resampled).
    7. Optional GIF generation for visual quality assurance.

Idempotency:
    If a final mapping JSON already exists for a series, the entire series is skipped, enabling
    safe re-runs without redundant computation.

Configurable Flags:
    NORMALIZATION_METHOD, TARGET_SPACING, ROW_COL_SWAPPED, FLIP_LR, MAKE_GIFS.
"""

import pandas as pd
import os
import json

from pathlib import Path
from tqdm import tqdm
from pipeline import convert_dicom_to_nii, strip_skull, correct_n4_bias, normalize, resample
from aneurysm_mapping import map_annotations_to_resampled
from utils import save_slices_as_gif


BASE_PATH = Path(__file__).resolve().parents[2]

DATA_DIR = os.path.join(BASE_PATH, "data")
TRAIN_PATH = os.path.join(DATA_DIR, "raw/mra_train.csv")
SERIES_DIR = os.path.join(DATA_DIR, "raw/mra_series")
PROCESSED_DIR = os.path.join(DATA_DIR, "preprocessed_data")
NII_DIR = os.path.join(PROCESSED_DIR, "nifti")
SKULL_STRIP_DIR = os.path.join(PROCESSED_DIR, "skull-strip")
GIF_DIR = os.path.join(PROCESSED_DIR, "gif")

# Create working / output directories if they do not exist (idempotent)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(NII_DIR, exist_ok=True)
os.makedirs(SKULL_STRIP_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)





df = pd.read_csv(TRAIN_PATH)
series_list = df["SeriesInstanceUID"].to_list()

### Batch pipeline execution with per-step output folders ###


should_execute = True  # set False for a dry-run listing only
list_to_process = series_list

# Directories for each pipeline stage
os.makedirs(NII_DIR, exist_ok=True)
os.makedirs(SKULL_STRIP_DIR, exist_ok=True)
N4_DIR = os.path.join(PROCESSED_DIR, "n4")
NORM_DIR = os.path.join(PROCESSED_DIR, "normalize")
RESAMPLE_DIR = os.path.join(PROCESSED_DIR, "resample")
MAPPING_DIR = os.path.join(PROCESSED_DIR, "mapping")

for d in (N4_DIR, NORM_DIR, RESAMPLE_DIR, MAPPING_DIR):
    os.makedirs(d, exist_ok=True)

CSV_LOCALIZERS = os.path.join(DATA_DIR, "raw/mra_train_localizers.csv")
if not os.path.isfile(CSV_LOCALIZERS):
    raise FileNotFoundError(f"Required annotations CSV missing: {CSV_LOCALIZERS}")
print(f"Annotations CSV: {CSV_LOCALIZERS}")

NORMALIZATION_METHOD = "minmax"
TARGET_SPACING = (1.0, 1.0, 1.0)  # set to None to skip resampling
ROW_COL_SWAPPED = True  # treat annotation x,y swapped (row/col) if True
FLIP_LR = True          # optional left-right flip when mapping annotations
MAKE_GIFS = False       # enable per-series GIF outputs

processed_series = []

if should_execute:
    # Skip a series only if its final mapping JSON already exists
    existing = {f.split('_mapping.json')[0] for f in os.listdir(MAPPING_DIR) if f.endswith('_mapping.json')}
    to_process = [sid for sid in list_to_process if sid not in existing]
    print(f"A traiter: {len(to_process)} series")

    for sid in tqdm(to_process, desc="Processing series", unit="series", dynamic_ncols=True):
        series_dir = os.path.join(SERIES_DIR, sid)
        try:
            # (1) DICOM -> NIfTI -------------------------------------------------
            nii_path = os.path.join(NII_DIR, sid + ".nii")
            if not os.path.isfile(nii_path):
                nii_path = convert_dicom_to_nii(series_dir, nii_path)
            else:
                print(f"[SKIP] Conversion already present for {sid}")

            # (2) Skull strip (SynthStrip) ---------------------------------------
            stripped_prefix = os.path.join(SKULL_STRIP_DIR, sid)
            stripped_img = stripped_prefix + "_output.nii.gz"
            if not os.path.isfile(stripped_img):
                stripped_img = strip_skull(nii_path, stripped_prefix)
            else:
                print(f"[SKIP] Skull strip already present for {sid}")
            mask_path = stripped_prefix + "_mask.nii.gz"
            if not os.path.isfile(mask_path):
                print(f"[WARN] Mask absent for {sid}")
                mask_path = None

            # (3) N4 bias field correction ---------------------------------------
            n4_path = os.path.join(N4_DIR, sid + "_n4.nii.gz")
            if not os.path.isfile(n4_path):
                n4_path = correct_n4_bias(stripped_img, out_path=n4_path, mask_path=mask_path)
            else:
                print(f"[SKIP] N4 already present for {sid}")

            # (4) Intensity normalization ----------------------------------------
            norm_path = os.path.join(NORM_DIR, sid + f"_{NORMALIZATION_METHOD}.nii.gz")
            if not os.path.isfile(norm_path):
                norm_path = normalize(n4_path, out_path=norm_path, mask_path=mask_path, method=NORMALIZATION_METHOD)
            else:
                print(f"[SKIP] Normalization already present for {sid}")

            # (5) Resample to target spacing --------------------------------------
            if TARGET_SPACING is not None:
                res_path = os.path.join(RESAMPLE_DIR, sid + "_res.nii.gz")
                if not os.path.isfile(res_path):
                    res_path = resample(norm_path, out_path=res_path, target_spacing=TARGET_SPACING)
                else:
                    print(f"[SKIP] Resample already present for {sid}")
            else:
                res_path = norm_path

            # (6) Map aneurysm annotation points into RESAMPLED space -------------
            mapped = map_annotations_to_resampled(
                csv_path=CSV_LOCALIZERS,
                series_dir=series_dir,
                original_nifti_path=nii_path,
                resampled_nifti_path=res_path,
                row_col_swapped=ROW_COL_SWAPPED,
                flip_lr=FLIP_LR,
                round_result=False,
                validate=True
            )
            print(f"{sid}: {len(mapped)} annotations mapped (resampled space)")

            json_entries = []
            for ann, vox in mapped:
                json_entries.append({
                    "SeriesInstanceUID": ann.series_uid,
                    "SOPInstanceUID": ann.sop_uid,
                    "location": ann.location,
                    "voxel": [float(v) for v in vox]
                })

            map_json_path = os.path.join(MAPPING_DIR, sid + "_mapping.json")
            with open(map_json_path, 'w', encoding='utf-8') as jf:
                json.dump(json_entries, jf, indent=2, ensure_ascii=False)
            print(f"Mapping JSON written: {map_json_path}")

            # (7) Optional GIF visualization --------------------------------------
            if MAKE_GIFS:
                raw_gif = os.path.join(GIF_DIR, sid + "_raw.gif")
                norm_gif = os.path.join(GIF_DIR, sid + "_norm.gif")
                res_gif = os.path.join(GIF_DIR, sid + "_res.gif")
                if not os.path.isfile(raw_gif):
                    save_slices_as_gif(nii_path, raw_gif, fps=40)
                if not os.path.isfile(norm_gif):
                    save_slices_as_gif(norm_path, norm_gif, fps=40)
                if TARGET_SPACING is not None and not os.path.isfile(res_gif):
                    save_slices_as_gif(res_path, res_gif, fps=40)

            processed_series.append(sid)
        except Exception as e:
            tqdm.write(f"[ERROR] Failed for {sid}: {e}")
            continue

    print(f"Done. {len(processed_series)} series processed.")
else:
    print("Pipeline batch not executed (should_execute=False). Set should_execute=True to run.")