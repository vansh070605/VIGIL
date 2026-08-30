import os
import glob
from PIL import Image
from torch.utils.data import Dataset

class MVTecVialDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, target_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g. dataset/)
            split (string): "train", "validation", "test_public", etc.
            transform (callable, optional): Optional transform to be applied on a sample.
            target_transform (callable, optional): Optional transform to be applied on the mask.
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = [] # 0 for normal, 1 for anomaly
        
        if split == "train":
            good_dir = os.path.join(root_dir, split, "good")
            paths = glob.glob(os.path.join(good_dir, "*.png"))
            self.image_paths.extend(paths)
            self.labels.extend([0] * len(paths))
            self.mask_paths.extend([None] * len(paths))
        elif split == "validation":
            good_dir = os.path.join(root_dir, split, "good")
            paths = glob.glob(os.path.join(good_dir, "*.png"))
            self.image_paths.extend(paths)
            self.labels.extend([0] * len(paths))
            self.mask_paths.extend([None] * len(paths))
        elif split == "test_public":
            # good images
            good_dir = os.path.join(root_dir, split, "good")
            good_paths = glob.glob(os.path.join(good_dir, "*.png"))
            self.image_paths.extend(good_paths)
            self.labels.extend([0] * len(good_paths))
            self.mask_paths.extend([None] * len(good_paths))
            
            # bad images
            bad_dir = os.path.join(root_dir, split, "bad")
            bad_paths = glob.glob(os.path.join(bad_dir, "*.png"))
            self.image_paths.extend(bad_paths)
            self.labels.extend([1] * len(bad_paths))
            
            # corresponding masks
            for bp in bad_paths:
                basename = os.path.basename(bp)
                mask_path = os.path.join(root_dir, split, "ground_truth", "bad", basename)
                self.mask_paths.append(mask_path)
                
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L') # MVTec AD Vial is grayscale
        
        label = self.labels[idx]
        mask_path = self.mask_paths[idx]
        
        if mask_path and os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('L')
        else:
            # Create a blank mask for normal images
            mask = Image.new('L', image.size, 0)
            
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
            
        return image, label, mask

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
from PIL import Image

@dataclass
class ImageSample:
    path: Path
    mask_path: Optional[Path]
    is_anomalous: bool
    variant: str
    index: str
    
    @property
    def label(self):
        return "bad" if self.is_anomalous else "good"
        
    def load_image(self, mode="RGB"):
        return np.array(Image.open(self.path).convert(mode))
        
    def load_mask(self):
        if not self.mask_path or not self.mask_path.exists():
            return None
        return np.array(Image.open(self.mask_path).convert("L"))

class MVTecAD2VialDataset:
    def __init__(self, root: str = None):
        self.root = Path(root) if root else Path("dataset")
        self._samples = {
            "train": [],
            "validation": [],
            "test_good": [],
            "test_bad": [],
            "test_private": [],
            "test_private_mixed": []
        }
        
    def get_split(self, split: str) -> List[ImageSample]:
        samples = []
        if split == "train":
            good_dir = self.root / "train" / "good"
            if good_dir.exists():
                for p in good_dir.glob("*.png"):
                    samples.append(ImageSample(
                        path=p,
                        mask_path=None,
                        is_anomalous=False,
                        variant="regular",
                        index=p.stem
                    ))
        elif split == "validation":
            good_dir = self.root / "validation" / "good"
            if good_dir.exists():
                for p in good_dir.glob("*.png"):
                    samples.append(ImageSample(
                        path=p,
                        mask_path=None,
                        is_anomalous=False,
                        variant="regular",
                        index=p.stem
                    ))
        elif split == "test_good":
            good_dir = self.root / "test_public" / "good"
            if good_dir.exists():
                for p in good_dir.glob("*.png"):
                    samples.append(ImageSample(
                        path=p,
                        mask_path=None,
                        is_anomalous=False,
                        variant="regular",
                        index=p.stem
                    ))
        elif split == "test_bad":
            bad_dir = self.root / "test_public" / "bad"
            if bad_dir.exists():
                for p in bad_dir.glob("*.png"):
                    parts = p.stem.split('_', 1)
                    variant = parts[1] if len(parts) > 1 else "regular"
                    mask_name = f"{p.stem}_mask.png"
                    samples.append(ImageSample(
                        path=p,
                        mask_path=self.root / "test_public" / "ground_truth" / "bad" / mask_name,
                        is_anomalous=True,
                        variant=variant,
                        index=p.stem
                    ))
        return samples

    def filter_by_variant(self, samples: List[ImageSample], variant: str) -> List[ImageSample]:
        return [s for s in samples if s.variant == variant]
        
    def validate_integrity(self):
        return {"errors": [], "warnings": []}
