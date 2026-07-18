import torch
import torch.nn.functional as F

def gram_matrix(feature_map):
    """
    Compute the Gram Matrix of a feature map.
    
    The Gram Matrix measures the correlation between different feature channels,
    which effectively captures the style (textures, colors, patterns) of an image.
    
    Args:
        feature_map (torch.Tensor): Input feature map of shape [B, C, H, W].
                                    (Though NST typically uses B=1, this handles batches).
    
    Returns:
        torch.Tensor: Gram matrix of shape [B, C, C].
    """
    B, C, H, W = feature_map.shape
    
    # Reshape to [B, C, H*W] so each channel is a flattened vector
    flattened = feature_map.view(B, C, H * W)
    
    # Compute Gram Matrix using batched matrix multiplication: [B, C, H*W] @ [B, H*W, C]
    # Resulting shape is [B, C, C]
    gram = (flattened  @ flattened.transpose(1, 2))
    
    # NORMALIZATION DECISION:
    # We normalize by dividing by (C * H * W). 
    # Why: In VGG19, early layers (like conv1_1) have huge spatial dimensions (H, W), 
    # leading to massively larger Gram values compared to deep layers (like conv5_1).
    # Dividing by the total number of elements (C * H * W) brings all layers to a 
    # relatively similar numerical scale, preventing early layers from dominating 
    # the style loss completely. This is the classic Gatys et al. (2015) approach.
    normalization_factor =  H * W
    gram = gram / normalization_factor
    
    return gram


def content_loss(content_target, generated):
    """
    Compute the content loss between target and generated feature maps.
    
    Args:
        content_target (torch.Tensor): Feature map from the content image, shape [B, C, H, W]
        generated (torch.Tensor): Feature map from the generated image, shape [B, C, H, W]
    
    Returns:
        torch.Tensor: Scalar MSE loss preserving the spatial structure.
    """
    # Simple Mean Squared Error. No Gram matrix needed here.
    return F.mse_loss(generated, content_target)


def style_loss(style_targets_dict, generated_dict, style_weights=None):
    """
    Compute the weighted style loss across multiple layers.
    
    Args:
        style_targets_dict (dict): Dictionary mapping layer names (str) to target 
                                   Gram matrices [B, C, C].
        generated_dict (dict): Dictionary mapping layer names (str) to generated 
                               feature maps [B, C, H, W] (will be converted to Gram inside).
        style_weights (dict, optional): Dictionary mapping layer names (str) to their 
                                        respective float weights. If None, equal weights 
                                        (1/N) are applied.
    
    Returns:
        torch.Tensor: Scalar total weighted style loss.
    """
    # Ensure we only compute loss for layers present in the target dict
    layers = style_targets_dict.keys()
    num_layers = len(layers)
    
    # Default to equal weights (e.g., 1/5 = 0.2 each) if none provided
    if style_weights is None:
        style_weights = {layer: 1.0 / num_layers for layer in layers}
        
    total_loss = torch.tensor(0.0, device=next(iter(generated_dict.values())).device)
    
    for layer in layers:
        # Get the precomputed Gram matrix for the style target
        target_gram = style_targets_dict[layer]
        
        # Compute the Gram matrix for the generated image's feature map
        gen_gram = gram_matrix(generated_dict[layer])
        
        # Compute MSE between the two Gram matrices
        layer_loss = F.mse_loss(gen_gram, target_gram)
        
        # Apply the specific weight for this layer and accumulate
        total_loss += (style_weights[layer] * layer_loss)
        
    return total_loss


if __name__ == "__main__":
    print("Running tests on losses.py...")
    
    # 1. Test Gram Matrix shape
    print("\n1. Testing gram_matrix...")
    B, C, H, W = 1, 512, 28, 28
    x = torch.randn(B, C, H, W)
    G = gram_matrix(x)
    assert G.shape == (B, C, C), f"Expected shape {(B, C, C)}, got {G.shape}"
    print(f"   SUCCESS: Gram matrix shape is correct -> {G.shape}")
    
    # 2. Test Content Loss backward pass
    print("\n2. Testing content_loss backward...")
    # requires_grad=True simulates the generated image needing gradients
    gen_feat = torch.randn(1, 512, 28, 28, requires_grad=True)
    target_feat = torch.randn(1, 512, 28, 28) # Target doesn't need grad
    
    c_loss = content_loss(target_feat, gen_feat)
    c_loss.backward()
    
    assert gen_feat.grad is not None, "Gradient is None, backward pass failed!"
    assert gen_feat.grad.abs().sum() > 0, "Gradient is all zeros!"
    print(f"   SUCCESS: Content loss backpropagated. Grad sum = {gen_feat.grad.abs().sum().item():.4f}")
    
    # 3. Test Style Loss (optional but good for completeness)
    print("\n3. Testing style_loss...")
    gen_dict = {
        'conv1_1': torch.randn(1, 64, 224, 224, requires_grad=True),
        'conv2_1': torch.randn(1, 128, 112, 112, requires_grad=True)
    }
    # Pre-compute target grams for the test
    target_grams = {k: gram_matrix(v.detach()) for k, v in gen_dict.items()}
    
    # Test with default equal weights
    s_loss = style_loss(target_grams, gen_dict)
    s_loss.backward()
    
    assert gen_dict['conv1_1'].grad is not None, "Style loss failed to backprop to conv1_1"
    assert gen_dict['conv2_1'].grad is not None, "Style loss failed to backprop to conv2_1"
    print(f"   SUCCESS: Style loss computed and backpropagated to all layers. Total loss = {s_loss.item():.4f}")
    
    print("\n=== All tests passed! ===")