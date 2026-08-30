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

def get_transform(resize=(256, 256)):
    return T.Compose([
        T.Resize(resize),
        T.ToTensor(),
    ])

def get_mask_transform(resize=(256, 256)):
    return T.Compose([
        T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
        T.ToTensor()
    ])
