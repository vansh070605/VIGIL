"""
VIGIL — Visualization utilities

Functions for displaying dataset samples, anomaly maps, and evaluation results.
All functions support both interactive display and saving to file.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data.dataset import ImageSample


def plot_samples_grid(
    samples: Sequence[ImageSample],
    ncols: int = 5,
    figsize: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Display a grid of image samples.

    Args:
        samples: List of ImageSample objects to display.
        ncols: Number of columns in the grid.
        figsize: Figure size (width, height) in inches.
        title: Optional figure title.
        save_path: If provided, save the figure to this path.
        show: If True, display the figure interactively.
    """
    n = len(samples)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (ncols * 4, nrows * 5)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
        axes = np.array(axes).reshape(1, -1)

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        if idx < n:
            img = samples[idx].load_image("L")
            ax.imshow(img, cmap="gray")
            label_color = "red" if samples[idx].is_anomalous else "green"
            label_text = "BAD" if samples[idx].is_anomalous else "GOOD"
            ax.set_title(
                f"[{label_text}] {samples[idx].path.name}",
                fontsize=8,
                color=label_color,
            )
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_mask_overlay(
    sample: ImageSample,
    alpha: float = 0.4,
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot an anomalous sample with its ground-truth mask overlay.

    Creates a three-panel figure: original, mask, overlay.

    Args:
        sample: An ImageSample with a mask_path.
        alpha: Transparency of the overlay.
        save_path: If provided, save the figure.
        show: If True, display interactively.
    """
    if sample.mask_path is None:
        raise ValueError(f"Sample {sample.path.name} has no mask_path")

    img = sample.load_image("RGB")
    mask = sample.load_mask()
    if mask is None:
        raise FileNotFoundError(f"Mask not found at {sample.mask_path}")

    # Create binary mask
    mask_binary = (mask > 127).astype(np.float32)

    # Build overlay
    overlay = img.copy().astype(np.float32)
    red = np.zeros_like(overlay)
    red[:, :, 0] = 255
    mask_3d = np.stack([mask_binary] * 3, axis=-1)
    overlay = overlay * (1 - mask_3d * alpha) + red * mask_3d * alpha
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Anomaly Mask Overlay: {sample.path.name}", fontsize=14, fontweight="bold")

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="hot")
    axes[1].set_title("Ground-Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_anomaly_map(
    image: np.ndarray,
    anomaly_map: np.ndarray,
    mask: Optional[np.ndarray] = None,
    title: str = "",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot a predicted anomaly heatmap against the original image.

    Args:
        image: Original image array (H, W) or (H, W, 3).
        anomaly_map: Predicted anomaly score map (H, W), float.
        mask: Optional ground-truth mask for comparison.
        title: Figure title.
        save_path: If provided, save the figure.
        show: If True, display interactively.
    """
    n_panels = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    cmap = "gray" if image.ndim == 2 else None
    axes[0].imshow(image, cmap=cmap)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    im = axes[1].imshow(anomaly_map, cmap="hot", interpolation="bilinear")
    axes[1].set_title("Anomaly Map (Predicted)")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if mask is not None:
        axes[2].imshow(mask, cmap="hot")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
