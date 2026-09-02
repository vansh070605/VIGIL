import os
import io
import yaml
import base64
from contextlib import asynccontextmanager

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from models.autoencoder import ConvAutoencoder
from preprocessing.transforms import get_transforms
import matplotlib.pyplot as plt

# Global state to hold the model
model = None
config = None
device = None
criterion = None
transform = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, config, device, criterion, transform
    
    # Load configuration
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")
    
    # Initialize Model
    model = ConvAutoencoder(
        in_channels=config["model"]["in_channels"], 
        latent_dim=config["model"]["latent_dim"]
    ).to(device)
    
    model_path = os.path.join(config["training"]["save_dir"], "best_model.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Successfully loaded pre-trained model weights.")
    else:
        print(f"Warning: Model weights not found at {model_path}. Using random initialization.")
        
    model.eval()
    criterion = nn.MSELoss(reduction='none')
    
    # We only need the forward transform
    transform, _ = get_transforms(image_size=tuple(config["data"]["image_size"]))
    
    yield
    
    # Clean up
    model = None

app = FastAPI(title="VIGIL API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictResponse(BaseModel):
    score: float
    is_anomalous: bool
    heatmap_base64: str

def generate_heatmap_base64(image_np, anomaly_map):
    # Resize anomaly map to match original image size
    img_h, img_w = image_np.shape[:2]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_np, cmap="gray")
    im = ax.imshow(anomaly_map, cmap="hot", interpolation="bilinear", alpha=0.5)
    ax.axis("off")
    
    # Save to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_b64}"

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # Read the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("L")
    original_np = np.array(image)
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        mse = criterion(output, input_tensor)
        
        # Calculate anomaly map and score
        anomaly_map = mse.mean(dim=1).squeeze().cpu().numpy()
        score = float(mse.mean().item())
        
    # Generate heatmap
    heatmap_b64 = generate_heatmap_base64(original_np, anomaly_map)
    
    # Threshold - in a real scenario this should be tuned on the validation set
    # Using a placeholder threshold of 0.05
    is_anomalous = score > 0.05
    
    return PredictResponse(
        score=score,
        is_anomalous=is_anomalous,
        heatmap_base64=heatmap_b64
    )
