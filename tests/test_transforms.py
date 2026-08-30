import torch
from PIL import Image
from src.preprocessing.transforms import get_transform, get_mask_transform

def test_get_transform():
    img = Image.new("RGB", (300, 400))
    transform = get_transform(resize=(224, 224))
    tensor = transform(img)
    
    assert tensor.shape == (3, 224, 224)

def test_get_mask_transform():
    img = Image.new("L", (300, 400))
    transform = get_mask_transform(resize=(224, 224))
    tensor = transform(img)
    
    assert tensor.shape == (1, 224, 224)
