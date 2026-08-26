"""
VIGIL — PatchCore Runner

Orchestrates the full PatchCore anomaly detection pipeline:
1. Load dataset
2. Create transforms and dataloaders
3. Extract training features and build memory bank
4. Run inference on test set
5. Compute evaluation metrics
6. Generate visualizations and save results

Usage:
    python -m src.models.runner
    python -m src.models.runner --resize 224 --coreset-ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MVTecAD2VialDataset
from src.data.torch_dataset import VialDataset
from src.evaluation.metrics import evaluate_full
from src.models.patchcore import PatchCore
from src.preprocessing.transforms import get_mask_transform, get_transform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vigil.runner")


def parse_args():
    parser = argparse.ArgumentParser(description="VIGIL PatchCore Runner")
    parser.add_argument(
        "--data-root", type=str, default=None,
        help="Path to dataset root (default: auto-detect)",
    )
    parser.add_argument(
        "--resize", type=int, default=256,
        help="Resize images to this size (default: 256)",
    )
    parser.add_argument(
        "--backbone", type=str, default="wide_resnet50_2",
        help="Backbone architecture (default: wide_resnet50_2)",
    )
    parser.add_argument(
        "--coreset-ratio", type=float, default=0.01,
        help="Coreset subsampling ratio (default: 0.01)",
    )
    parser.add_argument(
        "--num-neighbors", type=int, default=9,
        help="Number of nearest neighbors for scoring (default: 9)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for feature extraction (default: 4)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: experiments/patchcore_baseline)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def run_patchcore(args):
    """Execute the full PatchCore pipeline."""
    # ── Setup ──────────────────────────────────────────────────────────────
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "experiments" / "patchcore_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "anomaly_maps").mkdir(exist_ok=True)

    device = "cpu"  # No CUDA available in this environment
    logger.info("=" * 60)
    logger.info("VIGIL — PatchCore Baseline")
    logger.info("=" * 60)
    logger.info("Device: %s", device)
    logger.info("Resize: %d x %d", args.resize, args.resize)
    logger.info("Backbone: %s", args.backbone)
    logger.info("Coreset ratio: %.3f", args.coreset_ratio)
    logger.info("Num neighbors: %d", args.num_neighbors)
    logger.info("Output: %s", output_dir)

    # ── Load dataset ───────────────────────────────────────────────────────
    logger.info("")
    logger.info("Loading dataset...")
    data_root = Path(args.data_root) if args.data_root else None
    dataset = MVTecAD2VialDataset(root=data_root)
    logger.info("  %s", dataset)

    # ── Create transforms ──────────────────────────────────────────────────
    resize = (args.resize, args.resize)
    transform = get_transform(resize=resize)
    mask_transform = get_mask_transform(resize=resize)

    # ── Create dataloaders ─────────────────────────────────────────────────
    train_samples = dataset.get_split("train")
    test_good = dataset.get_split("test_good")
    test_bad = dataset.get_split("test_bad")
    test_samples = test_good + test_bad

    train_ds = VialDataset(train_samples, transform=transform)
    test_ds = VialDataset(
        test_samples, transform=transform, mask_transform=mask_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Windows compatibility
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    logger.info("  Train samples: %d", len(train_ds))
    logger.info("  Test samples: %d (good=%d, bad=%d)", len(test_ds), len(test_good), len(test_bad))

    # ── Build PatchCore model ──────────────────────────────────────────────
    model = PatchCore(
        backbone_name=args.backbone,
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=args.num_neighbors,
        device=device,
    )

    # ── Fit: extract features and build memory bank ────────────────────────
    logger.info("")
    logger.info("Phase 1: Building memory bank from training data...")
    t0 = time.time()
    model.fit(train_loader)
    fit_time = time.time() - t0
    logger.info("  Fit completed in %.1f seconds", fit_time)

    # Save model
    model_path = output_dir / "patchcore_model.pt"
    model.save(str(model_path))

    # ── Predict: run inference on test set ─────────────────────────────────
    logger.info("")
    logger.info("Phase 2: Running inference on test set...")
    t0 = time.time()
    image_scores, anomaly_maps, metadata = model.predict(test_loader)
    predict_time = time.time() - t0
    logger.info("  Inference completed in %.1f seconds", predict_time)

    # ── Collect ground truth ───────────────────────────────────────────────
    logger.info("")
    logger.info("Phase 3: Evaluating results...")

    labels = np.array([m["label"] for m in metadata])
    variants = [m["variant"] for m in metadata]

    # Collect masks from the test dataset
    gt_masks = []
    for i in range(len(test_ds)):
        batch = test_ds[i]
        # Mask is (1, H, W) tensor -> (H, W) numpy
        mask = batch["mask"].squeeze(0).numpy()
        gt_masks.append(mask)
    gt_masks = np.stack(gt_masks)

    # ── Evaluate ───────────────────────────────────────────────────────────
    results = evaluate_full(
        labels=labels,
        scores=image_scores,
        masks=gt_masks,
        anomaly_maps=anomaly_maps,
        variants=variants,
    )

    # ── Save results ───────────────────────────────────────────────────────
    results_dict = {
        "image_auroc": results.image_auroc,
        "image_f1": results.image_f1,
        "image_ap": results.image_ap,
        "image_threshold": results.image_threshold,
        "pixel_auroc": results.pixel_auroc,
        "pixel_f1": results.pixel_f1,
        "pixel_pro": results.pixel_pro,
        "pixel_threshold": results.pixel_threshold,
        "per_variant": results.per_variant,
        "config": {
            "resize": args.resize,
            "backbone": args.backbone,
            "coreset_ratio": args.coreset_ratio,
            "num_neighbors": args.num_neighbors,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "device": device,
            "n_train": len(train_samples),
            "n_test": len(test_samples),
            "fit_time_seconds": round(fit_time, 1),
            "predict_time_seconds": round(predict_time, 1),
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2, default=str)
    logger.info("Results saved to %s", output_dir / "results.json")

    # ── Generate visualizations ────────────────────────────────────────────
    logger.info("")
    logger.info("Phase 4: Generating visualizations...")
    _generate_visualizations(
        test_ds, image_scores, anomaly_maps, gt_masks, labels, metadata, results, output_dir
    )

    # ── Summary ────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("  Fit time:     %.1fs", fit_time)
    logger.info("  Predict time: %.1fs", predict_time)
    logger.info("  Image AUROC:  %.4f", results.image_auroc)
    logger.info("  Pixel AUROC:  %.4f", results.pixel_auroc)
    logger.info("  PRO:          %.4f", results.pixel_pro)
    logger.info("  Output:       %s", output_dir)

    return results


def _generate_visualizations(
    test_ds, image_scores, anomaly_maps, gt_masks, labels, metadata, results, output_dir
):
    """Generate and save all result visualizations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    from PIL import Image as PILImage

    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)

    # 1. Image-level ROC curve
    fpr, tpr, _ = roc_curve(labels, image_scores)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC = {results.image_auroc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Image-Level ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.savefig(viz_dir / "roc_curve_image.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved roc_curve_image.png")

    # 2. Anomaly map visualizations for bad samples
    bad_indices = [i for i, m in enumerate(metadata) if m["label"] == 1]
    # Pick up to 8 representative bad samples (regular variant preferred)
    regular_bad = [i for i in bad_indices if metadata[i]["variant"] == "regular"]
    show_indices = regular_bad[:8] if len(regular_bad) >= 8 else bad_indices[:8]

    n_show = len(show_indices)
    fig, axes = plt.subplots(n_show, 4, figsize=(20, 5 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle(
        "PatchCore Anomaly Detection Results",
        fontsize=18, fontweight="bold", y=1.02,
    )

    for row, idx in enumerate(show_indices):
        # Load original image
        sample = test_ds[idx]
        img_tensor = sample["image"]
        # Denormalize for display
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img_display = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        anomaly_map = anomaly_maps[idx]
        gt_mask = gt_masks[idx]
        score = image_scores[idx]

        # Original image
        axes[row, 0].imshow(img_display)
        axes[row, 0].set_title(f"Input (score={score:.3f})", fontsize=10)
        axes[row, 0].axis("off")

        # Anomaly map
        im = axes[row, 1].imshow(anomaly_map, cmap="hot", interpolation="bilinear")
        axes[row, 1].set_title("Anomaly Map", fontsize=10)
        axes[row, 1].axis("off")

        # Ground truth mask
        axes[row, 2].imshow(gt_mask, cmap="gray")
        axes[row, 2].set_title("Ground Truth", fontsize=10)
        axes[row, 2].axis("off")

        # Overlay
        overlay = img_display.copy()
        if anomaly_map.max() > 0:
            norm_map = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
            overlay[:, :, 0] = np.clip(overlay[:, :, 0] + norm_map * 0.5, 0, 1)
        axes[row, 3].imshow(overlay)
        axes[row, 3].set_title("Overlay", fontsize=10)
        axes[row, 3].axis("off")

    plt.tight_layout()
    fig.savefig(viz_dir / "anomaly_maps_bad.png", dpi=120, bbox_inches="tight")
    plt.close()
    logger.info("  Saved anomaly_maps_bad.png")

    # 3. Score distribution
    good_scores = image_scores[labels == 0]
    bad_scores = image_scores[labels == 1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(good_scores, bins=30, alpha=0.6, label=f"Normal (n={len(good_scores)})", color="green")
    ax.hist(bad_scores, bins=30, alpha=0.6, label=f"Anomalous (n={len(bad_scores)})", color="red")
    ax.axvline(results.image_threshold, color="blue", linestyle="--", linewidth=2,
               label=f"Threshold = {results.image_threshold:.3f}")
    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Score Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.savefig(viz_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved score_distribution.png")

    # 4. Per-variant bar chart
    if results.per_variant:
        variants = sorted(results.per_variant.keys())
        aurocs = [results.per_variant[v].get("image_auroc", 0) for v in variants]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        colors = ["#2ecc71" if a >= 0.9 else "#f39c12" if a >= 0.7 else "#e74c3c" for a in aurocs]
        bars = ax.bar(variants, aurocs, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Image AUROC", fontsize=12)
        ax.set_title("Per-Variant Image AUROC", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=results.image_auroc, color="blue", linestyle="--", alpha=0.5,
                    label=f"Overall: {results.image_auroc:.4f}")
        for bar, val in zip(bars, aurocs):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        fig.savefig(viz_dir / "per_variant_auroc.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("  Saved per_variant_auroc.png")

    logger.info("  All visualizations saved to %s", viz_dir)


if __name__ == "__main__":
    args = parse_args()
    run_patchcore(args)
