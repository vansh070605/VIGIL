import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from scipy.ndimage import label
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class EvaluationResult:
    image_auroc: float
    image_f1: float
    image_ap: float
    image_threshold: float
    pixel_auroc: float
    pixel_f1: float
    pixel_pro: float
    pixel_threshold: float
    per_variant: Dict[str, Any]

def calculate_auroc(y_true, y_score):
    if len(np.unique(y_true)) <= 1:
        return 0.5
    return roc_auc_score(y_true, y_score)

def calculate_pro(masks: np.ndarray, anomaly_maps: np.ndarray, threshold: float) -> float:
    """Calculate the Per-Region Overlap (PRO) metric."""
    pro_scores = []
    for i in range(len(masks)):
        mask = masks[i]
        amap = anomaly_maps[i]
        
        pred = amap > threshold
        labeled_mask, num_features = label(mask)
        
        if num_features == 0:
            continue
            
        for k in range(1, num_features + 1):
            component = (labeled_mask == k)
            overlap = np.logical_and(component, pred).sum() / component.sum()
            pro_scores.append(overlap)
            
    if not pro_scores:
        return 0.0
    return np.mean(pro_scores)

def evaluate_full(labels, scores, masks, anomaly_maps, variants) -> EvaluationResult:
    labels = np.array(labels)
    scores = np.array(scores)
    image_auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.5
    image_ap = average_precision_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0
    
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    image_f1 = f1_scores[best_idx]
    image_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(np.median(scores))
    
    masks_flat = masks.flatten() > 0
    maps_flat = anomaly_maps.flatten()
    
    if len(np.unique(masks_flat)) > 1:
        pixel_auroc = roc_auc_score(masks_flat, maps_flat)
        
        step = max(1, len(masks_flat) // 1000000)
        p_prec, p_rec, p_thresh = precision_recall_curve(masks_flat[::step], maps_flat[::step])
        p_f1s = 2 * (p_prec * p_rec) / (p_prec + p_rec + 1e-8)
        p_best_idx = np.argmax(p_f1s)
        pixel_f1 = float(p_f1s[p_best_idx])
        pixel_threshold = float(p_thresh[p_best_idx]) if p_best_idx < len(p_thresh) else float(np.median(maps_flat))
        
        pixel_pro = float(calculate_pro(masks, anomaly_maps, pixel_threshold))
    else:
        pixel_auroc = 0.5
        pixel_f1 = 0.0
        pixel_threshold = float(np.median(maps_flat))
        pixel_pro = 0.0
        
    per_variant = {}
    
    return EvaluationResult(
        image_auroc=float(image_auroc),
        image_f1=float(image_f1),
        image_ap=float(image_ap),
        image_threshold=image_threshold,
        pixel_auroc=float(pixel_auroc),
        pixel_f1=pixel_f1,
        pixel_pro=pixel_pro,
        pixel_threshold=pixel_threshold,
        per_variant=per_variant
    )
