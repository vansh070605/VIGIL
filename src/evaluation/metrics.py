"""
VIGIL — Evaluation Metrics for Anomaly Detection

Image-level metrics:
    - AUROC: Area Under the Receiver Operating Characteristic curve
    - F1 Score: at optimal threshold
    - Average Precision (AP): Area under precision-recall curve

Pixel-level metrics:
    - Pixel AUROC: pixel-wise AUROC
    - PRO: Per-Region Overlap (MVTec-standard metric)
    - Pixel F1: at optimal threshold

All metrics handle both numpy arrays and torch tensors as input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as ndimage_label
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for all evaluation metrics."""

    # Image-level
    image_auroc: float = 0.0
    image_f1: float = 0.0
    image_ap: float = 0.0
    image_threshold: float = 0.0

    # Pixel-level
    pixel_auroc: float = 0.0
    pixel_f1: float = 0.0
    pixel_pro: float = 0.0
    pixel_threshold: float = 0.0

    # Per-variant breakdown
    per_variant: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def summary(self) -> str:
        """Format results as a readable summary string."""
        lines = [
            "=" * 60,
            "EVALUATION RESULTS",
            "=" * 60,
            "",
            "Image-Level Metrics:",
            f"  AUROC:             {self.image_auroc:.4f}",
            f"  F1 Score:          {self.image_f1:.4f}",
            f"  Average Precision: {self.image_ap:.4f}",
            f"  Threshold:         {self.image_threshold:.4f}",
            "",
            "Pixel-Level Metrics:",
            f"  AUROC:             {self.pixel_auroc:.4f}",
            f"  F1 Score:          {self.pixel_f1:.4f}",
            f"  PRO:               {self.pixel_pro:.4f}",
            f"  Threshold:         {self.pixel_threshold:.4f}",
        ]

        if self.per_variant:
            lines.extend(["", "Per-Variant Image AUROC:"])
            for variant, metrics in sorted(self.per_variant.items()):
                auroc_val = metrics.get("image_auroc", float("nan"))
                n = metrics.get("n_samples", 0)
                lines.append(f"  {variant:15s}: {auroc_val:.4f}  (n={n})")

        lines.append("=" * 60)
        return "\n".join(lines)


def compute_image_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, float, float, float]:
    """Compute image-level anomaly detection metrics.

    Args:
        labels: Ground-truth labels, shape (N,). 0=normal, 1=anomalous.
        scores: Predicted anomaly scores, shape (N,). Higher = more anomalous.

    Returns:
        (auroc, f1, ap, optimal_threshold)
    """
    labels = np.asarray(labels, dtype=np.int32)
    scores = np.asarray(scores, dtype=np.float64)

    # AUROC
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        logger.warning("AUROC computation failed (only one class present?)")
        auroc = 0.0

    # Average Precision
    try:
        ap = average_precision_score(labels, scores)
    except ValueError:
        ap = 0.0

    # F1 at optimal threshold
    optimal_threshold, best_f1 = _find_optimal_threshold(labels, scores)

    return auroc, best_f1, ap, optimal_threshold


def compute_pixel_metrics(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    pro_integration_limit: float = 0.3,
) -> Tuple[float, float, float, float]:
    """Compute pixel-level anomaly segmentation metrics.

    Args:
        masks: Ground-truth binary masks, shape (N, H, W). Values in {0, 1}.
        anomaly_maps: Predicted anomaly score maps, shape (N, H, W).
        pro_integration_limit: FPR integration limit for PRO metric.

    Returns:
        (pixel_auroc, pixel_f1, pro, optimal_threshold)
    """
    masks = np.asarray(masks, dtype=np.int32)
    anomaly_maps = np.asarray(anomaly_maps, dtype=np.float64)

    # Flatten for pixel-level metrics
    masks_flat = masks.ravel()
    maps_flat = anomaly_maps.ravel()

    # Pixel AUROC
    try:
        pixel_auroc = roc_auc_score(masks_flat, maps_flat)
    except ValueError:
        logger.warning("Pixel AUROC failed")
        pixel_auroc = 0.0

    # Pixel F1 at optimal threshold
    optimal_threshold, pixel_f1 = _find_optimal_threshold(masks_flat, maps_flat)

    # PRO (Per-Region Overlap)
    pro = compute_pro(masks, anomaly_maps, integration_limit=pro_integration_limit)

    return pixel_auroc, pixel_f1, pro, optimal_threshold


def compute_pro(
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    integration_limit: float = 0.3,
    num_thresholds: int = 300,
) -> float:
    """Compute Per-Region Overlap (PRO) metric.

    PRO measures the average overlap between predicted anomaly regions
    and ground-truth regions, integrated over the false positive rate.

    This is the standard evaluation metric for MVTec AD.

    Args:
        masks: Ground-truth binary masks, shape (N, H, W).
        anomaly_maps: Predicted score maps, shape (N, H, W).
        integration_limit: Maximum FPR for integration.
        num_thresholds: Number of thresholds to evaluate.

    Returns:
        PRO score (normalized to [0, 1]).
    """
    # Collect all connected components across all masks
    # For each threshold, compute the overlap with each component
    all_fprs = []
    all_pros = []

    # Generate thresholds
    min_val = anomaly_maps.min()
    max_val = anomaly_maps.max()
    thresholds = np.linspace(max_val, min_val, num_thresholds)

    # Find all connected components in ground truth
    components = []
    for i in range(masks.shape[0]):
        if masks[i].max() == 0:
            continue  # Skip images without defects
        labeled, n_components = ndimage_label(masks[i])
        for comp_id in range(1, n_components + 1):
            comp_mask = (labeled == comp_id)
            components.append((i, comp_mask))

    if len(components) == 0:
        logger.warning("No defect regions found in masks for PRO computation")
        return 0.0

    # Total normal (non-defect) pixels for FPR computation
    total_normal_pixels = np.sum(masks == 0)
    if total_normal_pixels == 0:
        return 0.0

    for threshold in thresholds:
        # Binary prediction at this threshold
        predictions = (anomaly_maps >= threshold).astype(np.int32)

        # FPR: false positives among normal pixels
        fp = np.sum(predictions[masks == 0])
        fpr = fp / total_normal_pixels

        # Per-region overlap
        overlaps = []
        for img_idx, comp_mask in components:
            comp_pred = predictions[img_idx][comp_mask]
            overlap = comp_pred.sum() / comp_mask.sum()
            overlaps.append(overlap)

        mean_overlap = np.mean(overlaps)
        all_fprs.append(fpr)
        all_pros.append(mean_overlap)

    # Sort by FPR
    sorted_pairs = sorted(zip(all_fprs, all_pros))
    fprs = np.array([p[0] for p in sorted_pairs])
    pros = np.array([p[1] for p in sorted_pairs])

    # Integrate up to the limit
    mask = fprs <= integration_limit
    if mask.sum() < 2:
        return 0.0

    fprs_limited = fprs[mask]
    pros_limited = pros[mask]

    pro_auc = auc(fprs_limited, pros_limited)
    # Normalize by the integration limit
    pro_normalized = pro_auc / integration_limit

    return float(pro_normalized)


def _find_optimal_threshold(
    labels: np.ndarray, scores: np.ndarray
) -> Tuple[float, float]:
    """Find the threshold that maximizes F1 score.

    Args:
        labels: Binary ground-truth labels.
        scores: Predicted scores.

    Returns:
        (optimal_threshold, best_f1_score)
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, scores)

    # Compute F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_f1 = float(f1_scores[best_idx])

    if best_idx < len(thresholds):
        optimal_threshold = float(thresholds[best_idx])
    else:
        optimal_threshold = float(thresholds[-1]) if len(thresholds) > 0 else 0.0

    return optimal_threshold, best_f1


def evaluate_full(
    labels: np.ndarray,
    scores: np.ndarray,
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    variants: Optional[List[str]] = None,
) -> EvaluationResults:
    """Run the complete evaluation pipeline.

    Args:
        labels: Image-level labels, shape (N,). 0=normal, 1=anomalous.
        scores: Image-level anomaly scores, shape (N,).
        masks: Pixel-level ground-truth masks, shape (N, H, W).
        anomaly_maps: Pixel-level anomaly score maps, shape (N, H, W).
        variants: Optional list of variant names for per-variant evaluation.

    Returns:
        EvaluationResults with all metrics populated.
    """
    results = EvaluationResults()

    # Image-level
    auroc, f1, ap, threshold = compute_image_metrics(labels, scores)
    results.image_auroc = auroc
    results.image_f1 = f1
    results.image_ap = ap
    results.image_threshold = threshold

    # Pixel-level
    pixel_auroc, pixel_f1, pro, pixel_threshold = compute_pixel_metrics(
        masks, anomaly_maps
    )
    results.pixel_auroc = pixel_auroc
    results.pixel_f1 = pixel_f1
    results.pixel_pro = pro
    results.pixel_threshold = pixel_threshold

    # Per-variant breakdown
    if variants is not None:
        unique_variants = sorted(set(variants))
        for variant in unique_variants:
            var_mask = np.array([v == variant for v in variants])
            var_labels = labels[var_mask]
            var_scores = scores[var_mask]

            # Only compute AUROC if both classes are present
            if len(set(var_labels)) >= 2:
                var_auroc, var_f1, var_ap, _ = compute_image_metrics(
                    var_labels, var_scores
                )
            else:
                var_auroc = float("nan")
                var_f1 = float("nan")
                var_ap = float("nan")

            results.per_variant[variant] = {
                "image_auroc": var_auroc,
                "image_f1": var_f1,
                "image_ap": var_ap,
                "n_samples": int(var_mask.sum()),
            }

    logger.info("\n%s", results.summary())
    return results
