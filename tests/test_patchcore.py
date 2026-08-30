import torch
from src.models.patchcore import PatchCore

def test_coreset_subsample():
    torch.manual_seed(42)
    patches = torch.randn(100, 16)
    
    model = PatchCore(coreset_sampling_ratio=0.1, device="cpu")
    subsampled = model._coreset_subsample(patches)
    
    assert subsampled.shape[0] == 10
    assert subsampled.shape[1] == 16
