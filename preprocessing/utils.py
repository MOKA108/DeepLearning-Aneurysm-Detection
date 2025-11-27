"""Visualization utilities for quick inspection of NIfTI volumes.

Functions:
    load_nifti_as_zyx: Load a NIfTI file, return 3D array shaped `(Z, H, W)`.
    to_uint8: Rescale intensities into 0–255 `uint8` for display/GIF export.
    play_slices_like_gif: Animate axial slices in a loop using Matplotlib.
    save_slices_as_gif: Persist an animated axial slice GIF to disk.

Notes:
    - `radiological=True` flips left/right to match common neuroimaging conventions.
    - Use `max_side` to downscale frames for smaller GIF file sizes.
    - All helpers are convenience only; they do not alter training data logic.
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio.v2 as imageio
from pathlib import Path
from PIL import Image
import os

def load_nifti_as_zyx(nifti_path: str) -> np.ndarray:
    """Load a NIfTI image and return a `(Z, H, W)` array.

    If the image is 4D, only the first volume is retained.

    Args:
        nifti_path: Path to NIfTI file.
    Returns:
        np.ndarray: 3D float32 volume shaped `(Z, H, W)`.
    """
    img = nib.load(nifti_path)
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    # Many NIfTI files are (H, W, Z); we treat data as (Z, H, W) unless manual adjustment is needed.
    # Simple heuristic: if first dimension is much smaller than the last it may already be (Z, H, W).
    # We avoid implicit transposition to prevent surprises. Adjust manually if needed:
    # For volumes stored as (H, W, Z) you can do: data = np.moveaxis(data, -1, 0)
    return data.astype(np.float32, copy=False)

def to_uint8(vol: np.ndarray, vmin=None, vmax=None) -> np.ndarray:
    """Convert a volume to `uint8` with optional percentile-based clipping.

    Args:
        vol: Input numeric volume.
        vmin: Lower bound (auto 1st percentile if None).
        vmax: Upper bound (auto 99th percentile if None).
    Returns:
        np.ndarray: `uint8` volume (same shape as input).
    """
    if vmin is None:
        vmin = float(np.percentile(vol, 1))
    if vmax is None:
        vmax = float(np.percentile(vol, 99))
    vol = np.clip(vol, vmin, vmax)
    vol = (vol - vmin) / (vmax - vmin + 1e-6)
    return (vol * 255).astype(np.uint8)

def play_slices_like_gif(nifti_path: str, interval_ms: int = 50, radiological: bool = True):
    """Animate axial slices of a volume in a Matplotlib window.

    Args:
        nifti_path: Path to NIfTI image.
        interval_ms: Delay between frames in milliseconds.
        radiological: If True, flip left/right for radiological orientation.
    """
    vol = load_nifti_as_zyx(nifti_path)         # (Z,H,W)
    vol8 = to_uint8(vol)                         # (Z,H,W) uint8
    if radiological:
        vol8 = np.flip(vol8, axis=2)             # left/right flip for radiological view

    Z = vol8.shape[0]

    fig, ax = plt.subplots()
    im = ax.imshow(vol8[0], cmap="gray", animated=True)
    ax.axis("off")
    title = ax.set_title(f"Slice 1/{Z}")

    def update(i):
        im.set_array(vol8[i])
        title.set_text(f"Slice {i+1}/{Z}")
        return (im,)

    anim = FuncAnimation(fig, update, frames=Z, interval=interval_ms, blit=True, repeat=True)
    plt.show()


def save_slices_as_gif(
    nifti_path: str,
    out_gif: str,
    fps: int = 15,
    radiological: bool = True,
    max_side: int = 512,
) -> str:
    """Export a looping axial-slice GIF for quick visual QA.

    Args:
        nifti_path: Path to source NIfTI image.
        out_gif: Destination GIF path (extension added if missing).
        fps: Frames per second for playback speed.
        radiological: If True, apply left/right flip.
        max_side: Maximum pixel size for height or width (downscale if larger).

    Returns:
        str: Final GIF file path.
    """

    # Force .gif extension to avoid using TIFF writer (no duration support)
    if not out_gif.lower().endswith('.gif'):
        out_gif = out_gif + '.gif'
    # Crée le dossier cible si nécessaire
    os.makedirs(os.path.dirname(out_gif), exist_ok=True)

    vol = load_nifti_as_zyx(nifti_path)
    vol8 = to_uint8(vol)

    if radiological:
        vol8 = np.flip(vol8, axis=2)

    Z, H, W = vol8.shape
    scale = min(1.0, max_side / max(H, W))
    frames = []
    for z in range(Z):
        frame = vol8[z]
        if scale < 1.0:
            new_size = (int(W * scale), int(H * scale))
            frame = np.array(Image.fromarray(frame).resize(new_size, Image.BILINEAR))
        frames.append(frame)

    duration = 1.0 / max(fps, 1)
    try:
        # imageio v2 GIF writer supporte duration
        imageio.mimsave(out_gif, frames, duration=duration, format='GIF')
    except TypeError:
        # Fallback: construire via PIL pour compatibilité
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            out_gif,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(duration * 1000),  # ms
            loop=0
        )
    return out_gif