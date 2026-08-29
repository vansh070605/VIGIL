import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from data.dataset import MVTecVialDataset
from preprocessing.transforms import get_transforms
from models.autoencoder import ConvAutoencoder
from evaluation.metrics import calculate_auroc

def train():
    # Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup data
    transform, target_transform = get_transforms(image_size=tuple(config["data"]["image_size"]))
    
    train_dataset = MVTecVialDataset(
        root_dir=config["data"]["dataset_path"], 
        split="train", 
        transform=transform,
        target_transform=target_transform
    )
    
    val_dataset = MVTecVialDataset(
        root_dir=config["data"]["dataset_path"], 
        split="test_public", 
        transform=transform,
        target_transform=target_transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=True, 
        num_workers=config["data"]["num_workers"]
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["data"]["batch_size"], 
        shuffle=False, 
        num_workers=config["data"]["num_workers"]
    )
    
    # Setup model
    model = ConvAutoencoder(
        in_channels=config["model"]["in_channels"], 
        latent_dim=config["model"]["latent_dim"]
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config["training"]["learning_rate"], 
        weight_decay=config["training"]["weight_decay"]
    )
    
    save_dir = config["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    best_auroc = 0.0
    
    # Training Loop
    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            images = images.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                images = images.to(device)
                outputs = model(images)
                
                # Image-level anomaly score is the MSE error per image
                mse = nn.MSELoss(reduction='none')(outputs, images)
                anomaly_scores = mse.view(mse.size(0), -1).mean(dim=1).cpu().numpy()
                
                val_loss += mse.mean().item() * images.size(0)
                
                all_labels.extend(labels.numpy())
                all_scores.extend(anomaly_scores)
                
        val_loss /= len(val_loader.dataset)
        
        image_auroc = calculate_auroc(np.array(all_labels), np.array(all_scores))
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val AUROC: {image_auroc:.4f}")
        
        if image_auroc > best_auroc:
            best_auroc = image_auroc
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print(f"--> Saved new best model with AUROC: {best_auroc:.4f}")

if __name__ == "__main__":
    train()
