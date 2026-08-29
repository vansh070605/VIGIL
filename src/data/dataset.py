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
