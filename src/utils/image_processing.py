import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# ImageNet normalization constants for PyTorch VGG (RGB order, 0-1 range)
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB order
IMAGENET_STD = [0.229, 0.224, 0.225]   # RGB order


def load_image(image_path, max_size=512):
    """
    Load image and resize keeping aspect ratio with max side = max_size.
    
    Args:
        image_path: Path to image file
        max_size: Maximum dimension (width or height)
    
    Returns:
        PIL Image object in RGB mode
    """
    img = Image.open(image_path).convert('RGB')
    
    # Resize keeping aspect ratio
    width, height = img.size
    if max(width, height) > max_size:
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return img


def preprocess_vgg(image, target_size=None):
    """
    Preprocess image for PyTorch VGG network.
    
    Pipeline:
    1. PIL Image (0-255, RGB) 
    2. Resize to target_size (optional)
    3. Convert to Tensor (0-1 range, RGB)
    4. Normalize with ImageNet mean/std
    
    Args:
        image: PIL Image in RGB mode
        target_size: (height, width) tuple to resize to (optional)
    
    Returns:
        torch.FloatTensor of shape [1, 3, H, W] ready for VGG
    """
    # Step 1: Optional resizing
    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Step 2: Convert to tensor and scale to 0-1
    # transforms.ToTensor() does: PIL (0-255) -> FloatTensor (0-1) and [H,W,C] -> [C,H,W]
    img_tensor = transforms.ToTensor()(image)
    # Shape: [3, H, W], values in range [0, 1]
    
    # Step 3: Normalize with ImageNet stats (RGB order)
    # This does: (x - mean) / std for each channel
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    img_tensor = normalize(img_tensor)
    # Shape: [3, H, W], values approximately in range [-2, 2]
    
    # Step 4: Add batch dimension
    return img_tensor.unsqueeze(0)  # [1, 3, H, W]


def deprocess_vgg(tensor):
    """
    Reverse VGG preprocessing to get displayable PIL image.
    
    Reverses:
    1. Remove batch dimension
    2. Unnormalize: (x * std) + mean
    3. Clamp to [0, 1]
    4. Convert to PIL (0-255)
    
    Args:
        tensor: torch.FloatTensor [1, 3, H, W] from VGG
    
    Returns:
        PIL Image in RGB format
    """
    # Step 1: Remove batch dimension and detach
    img = tensor.squeeze(0).cpu().detach()
    # Shape: [3, H, W]
    
    # Step 2: Unnormalize: (x * std) + mean
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    img = img * std + mean
    # Shape: [3, H, W], values in range roughly [0, 1]
    
    # Step 3: Clamp to valid range [0, 1]
    img = torch.clamp(img, 0, 1)
    
    # Step 4: Convert to PIL (0-255)
    # transforms.ToPILImage() expects [C, H, W] in [0, 1] range
    to_pil = transforms.ToPILImage()
    return to_pil(img)


def preprocess_vgg_manual(image, target_size=None):
    """
    Manual version of preprocessing for educational purposes.
    Same as preprocess_vgg but with explicit steps.
    
    Args:
        image: PIL Image in RGB mode
        target_size: (height, width) tuple
    
    Returns:
        torch.FloatTensor [1, 3, H, W]
    """
    # Step 1: Resize
    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # Step 2: Convert PIL to numpy (0-255)
    img_array = np.array(image).astype(np.float32)
    # Shape: [H, W, 3], values 0-255
    
    # Step 3: Scale to 0-1
    img_array = img_array / 255.0
    # Shape: [H, W, 3], values 0-1
    
    # Step 4: Convert to tensor [C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
    # Shape: [3, H, W]
    
    # Step 5: Normalize with ImageNet stats (RGB order)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    # Step 6: Add batch dimension
    return img_tensor.unsqueeze(0)  # [1, 3, H, W]


def deprocess_vgg_manual(tensor):
    """
    Manual version of deprocessing for educational purposes.
    
    Args:
        tensor: torch.FloatTensor [1, 3, H, W]
    
    Returns:
        PIL Image in RGB
    """
    # Step 1: Remove batch dimension and detach
    img = tensor.squeeze(0).cpu().detach()
    
    # Step 2: Unnormalize: (x * std) + mean
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    img = img * std + mean
    
    # Step 3: Clamp and convert to numpy
    img = torch.clamp(img, 0, 1)
    img_np = img.permute(1, 2, 0).numpy()  # [H, W, 3]
    img_np = (img_np * 255).astype(np.uint8)  # 0-255
    
    return Image.fromarray(img_np)


def resize_keep_aspect(image, max_side):
    """
    Resize PIL image keeping aspect ratio with max side = max_side.
    
    Args:
        image: PIL Image
        max_side: Maximum dimension
    
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    if max(width, height) <= max_side:
        return image
    
    scale = max_side / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def get_image_size(image_path):
    """Get (width, height) of image without loading fully."""
    with Image.open(image_path) as img:
        return img.size


def get_vgg_preprocessing_transform(target_size=None):
    """
    Get a torchvision transform pipeline for VGG preprocessing.
    Useful for composing with other transforms.
    
    Args:
        target_size: (height, width) tuple or None
    
    Returns:
        torchvision.transforms.Compose pipeline
    """
    transforms_list = []
    
    if target_size:
        transforms_list.append(transforms.Resize(target_size, Image.Resampling.LANCZOS))
    
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return transforms.Compose(transforms_list)


# Quick test function
def test_preprocessing():
    """Test that preprocessing and deprocessing work correctly."""
    # Create a test image
    test_img = Image.new('RGB', (100, 100), color='red')
    
    # Preprocess
    tensor = preprocess_vgg(test_img)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor min: {tensor.min():.3f}, max: {tensor.max():.3f}")
    print(f"Tensor mean per channel: {tensor.mean(dim=[2,3]).squeeze()}")
    
    # Deprocess
    recovered = deprocess_vgg(tensor)
    print(f"Recovered size: {recovered.size}")
    
    # Compare (should be close)
    orig_array = np.array(test_img)
    rec_array = np.array(recovered)
    diff = np.abs(orig_array.astype(float) - rec_array.astype(float))
    print(f"Mean difference: {diff.mean():.2f}")
    
    return tensor, recovered


if __name__ == "__main__":
    test_preprocessing()