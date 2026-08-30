"""
VIGIL — PatchCore Anomaly Detection Model

Implementation of PatchCore (Roth et al., 2022) for unsupervised anomaly
detection and localization.

Algorithm overview:
1. Extract patch-level features from intermediate layers of a pretrained CNN
2. Build a memory bank of normal patch features from training data
3. Apply coreset subsampling to reduce memory bank size
4. At inference, compute anomaly score as distance to nearest normal patch

Reference:
    Roth, K., et al. "Towards Total Recall in Industrial Anomaly Detection."
    CVPR 2022. https://arxiv.org/abs/2106.08265
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights

logger = logging.getLogger(__name__)


class FeatureExtractor(torch.nn.Module):
    """Extract intermediate features from a pretrained backbone.

    Registers forward hooks on specified layers to capture their outputs
    during the forward pass.

    Args:
        backbone_name: Currently only 'wide_resnet50_2' is supported.
        layers: List of layer names to extract features from.
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = None,
    ):
        super().__init__()

        if layers is None:
            layers = ["layer2", "layer3"]
        self.layers = layers

        # Load pretrained backbone
        if backbone_name == "wide_resnet50_2":
            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            self.backbone = wide_resnet50_2(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # Storage for hooked features
        self._features: Dict[str, torch.Tensor] = {}

        # Register hooks
        for layer_name in self.layers:
            layer = dict(self.backbone.named_modules())[layer_name]
            layer.register_forward_hook(self._make_hook(layer_name))

    def _make_hook(self, layer_name: str):
        """Create a forward hook that captures layer output."""
        def hook(module, input, output):
            self._features[layer_name] = output
        return hook

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract features from the specified layers.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Dictionary mapping layer names to feature tensors.
        """
        self._features = {}
        _ = self.backbone(x)
        return dict(self._features)


class PatchCore:
    """PatchCore anomaly detection model.

    This class implements the full PatchCore pipeline:
    - Feature extraction from a pretrained CNN
    - Memory bank construction from normal training patches
    - Coreset subsampling for efficiency
    - Anomaly scoring via k-NN distance

    Args:
        backbone_name: Name of the pretrained backbone.
        layers: Layers to extract features from.
        coreset_sampling_ratio: Fraction of patches to keep in memory bank.
        num_neighbors: Number of nearest neighbors for anomaly scoring.
        device: Torch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: Optional[List[str]] = None,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        device: str = "cpu",
    ):
        if layers is None:
            layers = ["layer2", "layer3"]

        self.device = torch.device(device)
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.layers = layers

        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(backbone_name, layers)
        self.feature_extractor.to(self.device)

        # Memory bank (populated during fit)
        self.memory_bank: Optional[torch.Tensor] = None

        # Spatial dimensions of the feature map (set during fit)
        self._feature_map_shape: Optional[Tuple[int, int]] = None

    def _combine_features(
        self, features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Combine multi-layer features into a single patch feature map.

        Resizes all feature maps to the largest spatial resolution among
        the extracted layers, then concatenates along the channel dimension.

        Args:
            features: Dict of layer_name -> (B, C, H, W) tensors.

        Returns:
            Combined feature tensor of shape (B, C_total, H_max, W_max).
        """
        layer_features = [features[l] for l in self.layers]

        # Use the spatial size of the first (largest) layer
        target_size = layer_features[0].shape[2:]

        aligned = []
        for feat in layer_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(
                    feat, size=target_size, mode="bilinear", align_corners=False
                )
            aligned.append(feat)

        # Concatenate along channel dimension: (B, C1+C2, H, W)
        return torch.cat(aligned, dim=1)

    def _reshape_to_patches(self, features: torch.Tensor) -> torch.Tensor:
        """Reshape spatial feature map to a list of patch feature vectors.

        Args:
            features: Tensor of shape (B, C, H, W).

        Returns:
            Tensor of shape (B * H * W, C) — one vector per patch.
        """
        B, C, H, W = features.shape
        # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        patches = features.permute(0, 2, 3, 1).reshape(-1, C)
        return patches

    def _extract_patch_features(
        self, dataloader: DataLoader
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Extract patch features from all images in a dataloader.

        Args:
            dataloader: DataLoader yielding dicts with 'image' key.

        Returns:
            (all_patches, (H, W)) where all_patches has shape (N_total, C)
            and (H, W) is the spatial shape of the feature map.
        """
        all_patches = []
        feature_map_shape = None

        self.feature_extractor.eval()
        total = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(self.device)
            features = self.feature_extractor(images)
            combined = self._combine_features(features)

            if feature_map_shape is None:
                feature_map_shape = combined.shape[2:]  # (H, W)

            patches = self._reshape_to_patches(combined)
            all_patches.append(patches.cpu())

            if (batch_idx + 1) % 10 == 0 or batch_idx == total - 1:
                logger.info(
                    "  Feature extraction: batch %d/%d", batch_idx + 1, total
                )

        all_patches = torch.cat(all_patches, dim=0)
        return all_patches, feature_map_shape

    def _coreset_subsample(self, patches: torch.Tensor) -> torch.Tensor:
        """Subsample patches using Greedy Coreset Selection (Farthest Point Sampling).
        
        Args:
            patches: Tensor of shape (N, C).

        Returns:
            Subsampled tensor of shape (M, C) where M = N * ratio.
        """
        n_total = patches.shape[0]
        n_select = max(1, int(n_total * self.coreset_sampling_ratio))

        if n_select >= n_total:
            logger.info("  Coreset: keeping all %d patches (ratio >= 1)", n_total)
            return patches

        logger.info(
            "  Farthest Point Sampling: %d -> %d patches (%.1f%%)",
            n_total, n_select, self.coreset_sampling_ratio * 100,
        )
        
        device = patches.device
        
        # Start with a random index
        coreset_idx = [torch.randint(0, n_total, (1,)).item()]
        
        # Keep track of the minimum distance from each point to the selected coreset points
        # Initial distances from all points to the first selected point
        min_distances = torch.cdist(patches, patches[coreset_idx[0]:coreset_idx[0]+1]).squeeze()
        
        for i in range(1, n_select):
            if i % 100 == 0:
                logger.info("    Selected %d/%d points...", i, n_select)
                
            # Select the point that has the maximum minimum distance to the coreset
            max_idx = torch.argmax(min_distances).item()
            coreset_idx.append(max_idx)
            
            # Update min distances for the next iteration
            new_dist = torch.cdist(patches, patches[max_idx:max_idx+1]).squeeze()
            min_distances = torch.minimum(min_distances, new_dist)
            
        return patches[coreset_idx]

    @staticmethod
    def _chunked_distances(
        a: torch.Tensor, b: torch.Tensor, chunk_size: int = 10000
    ) -> torch.Tensor:
        """Compute L2 distances between each row of a and each row of b.

        Processes in chunks to manage memory on CPU.

        Args:
            a: Tensor of shape (N, C).
            b: Tensor of shape (M, C).
            chunk_size: Number of rows of a to process at once.

        Returns:
            Distance matrix of shape (N, M).
        """
        distances = []
        for start in range(0, a.shape[0], chunk_size):
            end = min(start + chunk_size, a.shape[0])
            chunk = a[start:end]
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a*b^T
            dist = torch.cdist(chunk, b)
            distances.append(dist)
        return torch.cat(distances, dim=0)

    def fit(self, dataloader: DataLoader) -> None:
        """Build the memory bank from normal training images.

        Args:
            dataloader: DataLoader of normal training images.
        """
        logger.info("PatchCore fit: extracting features from training data...")
        patches, self._feature_map_shape = self._extract_patch_features(dataloader)
        logger.info(
            "  Extracted %d patches, feature dim=%d, spatial=%s",
            patches.shape[0], patches.shape[1], self._feature_map_shape,
        )

        # Coreset subsampling
        self.memory_bank = self._coreset_subsample(patches)
        logger.info(
            "  Memory bank size: %d patches (%d dims)",
            self.memory_bank.shape[0], self.memory_bank.shape[1],
        )

    @torch.no_grad()
    def predict(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """Run anomaly detection on a test dataloader.

        Args:
            dataloader: DataLoader of test images.

        Returns:
            (image_scores, anomaly_maps, metadata_list) where:
            - image_scores: (N_images,) array of image-level anomaly scores
            - anomaly_maps: (N_images, H_orig, W_orig) array of pixel-level scores
            - metadata_list: list of dicts with path, label, variant, index
        """
        if self.memory_bank is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        all_scores = []
        all_maps = []
        all_meta = []

        self.feature_extractor.eval()
        total = len(dataloader)

        for batch_idx, batch in enumerate(dataloader):
            images = batch["image"].to(self.device)
            B = images.shape[0]

            # Extract features
            features = self.feature_extractor(images)
            combined = self._combine_features(features)
            fH, fW = combined.shape[2:]

            # Reshape to patches: (B*fH*fW, C)
            patches = self._reshape_to_patches(combined).cpu()

            # Compute distances to k nearest neighbors in memory bank
            # Process per-image to manage memory
            for i in range(B):
                img_patches = patches[i * fH * fW : (i + 1) * fH * fW]

                # Compute distances to all memory bank entries
                distances = self._chunked_distances(img_patches, self.memory_bank)

                # k-NN: take the k-th nearest distance for each patch
                k = min(self.num_neighbors, self.memory_bank.shape[0])
                topk_distances, _ = torch.topk(distances, k, dim=1, largest=False)

                # Patch-level anomaly score = mean of k-NN distances
                patch_scores = topk_distances.mean(dim=1)

                # Reshape to spatial map
                score_map = patch_scores.reshape(fH, fW).numpy()

                # Upsample to input image resolution
                score_map_tensor = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0)
                input_h, input_w = images.shape[2], images.shape[3]
                score_map_upsampled = F.interpolate(
                    score_map_tensor.float(),
                    size=(input_h, input_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze().numpy()

                # Apply Gaussian smoothing for cleaner anomaly maps
                from scipy.ndimage import gaussian_filter
                score_map_upsampled = gaussian_filter(score_map_upsampled, sigma=4)

                # Image-level score = max patch score
                image_score = float(patch_scores.max())

                all_scores.append(image_score)
                all_maps.append(score_map_upsampled)

            # Collect metadata
            for i in range(B):
                all_meta.append({
                    "path": batch["path"][i],
                    "label": batch["label"][i].item(),
                    "variant": batch["variant"][i],
                    "index": batch["index"][i],
                })

            if (batch_idx + 1) % 5 == 0 or batch_idx == total - 1:
                logger.info(
                    "  Inference: batch %d/%d", batch_idx + 1, total
                )

        image_scores = np.array(all_scores)
        anomaly_maps = np.stack(all_maps)

        return image_scores, anomaly_maps, all_meta

    def save(self, path: str) -> None:
        """Save the fitted memory bank to disk."""
        if self.memory_bank is None:
            raise RuntimeError("Model not fitted. Nothing to save.")
        state = {
            "memory_bank": self.memory_bank,
            "feature_map_shape": self._feature_map_shape,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "num_neighbors": self.num_neighbors,
            "layers": self.layers,
        }
        torch.save(state, path)
        logger.info("Model saved to %s", path)

    def load(self, path: str) -> None:
        """Load a previously fitted memory bank from disk."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.memory_bank = state["memory_bank"]
        self._feature_map_shape = state["feature_map_shape"]
        self.coreset_sampling_ratio = state["coreset_sampling_ratio"]
        self.num_neighbors = state["num_neighbors"]
        self.layers = state["layers"]
        logger.info(
            "Model loaded from %s (memory bank: %d patches)",
            path, self.memory_bank.shape[0],
        )
