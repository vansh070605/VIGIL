import torchvision.transforms as T

def get_transforms(image_size=(256, 256)):
    """
    Returns the torchvision transforms for the dataset.
    """
    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
    ])
    
    target_transform = T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor()
    ])
    
    return transform, target_transform
