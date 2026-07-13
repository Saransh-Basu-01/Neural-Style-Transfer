import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGG19FeatureExtractor:
    """
    VGG19 feature extractor that loads pretrained weights, freezes all parameters,
    and extracts feature maps from specified layers.
    """
    
    def __init__(self, content_layer='conv4_2', style_layers=None):
        """
        Initialize VGG19 feature extractor.
        
        Args:
            content_layer (str): Layer name for content extraction (default: 'conv4_2')
            style_layers (list): List of layer names for style extraction 
                                (default: ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'])
        """
        # Load pretrained VGG19
        self.model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Set to eval mode
        self.model.eval()
        
        # Set content and style layers
        self.content_layer = content_layer
        if style_layers is None:
            self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
        else:
            self.style_layers = style_layers
        
        # Combine all layers to extract
        self.layers = list(set([content_layer] + self.style_layers))
        
        # Map layer names to indices
        self.layer_name_to_index = self._create_layer_mapping()
        self.idx_to_name = {v: k for k, v in self.layer_name_to_index.items() if k in self.layers}
        # Create a list of layer indices to extract
        self.layer_indices = []
        for layer_name in self.layers:
            if layer_name in self.layer_name_to_index:
                self.layer_indices.append(self.layer_name_to_index[layer_name])
            else:
                raise ValueError(f"Layer '{layer_name}' not found in VGG19")
    
    def _create_layer_mapping(self):
        """
        Create mapping from layer names to CORRECT indices in VGG19 features.
        
        In torchvision VGG19, features has 36 layers:
        0:  conv1_1
        1:  relu1_1   ← We extract AFTER ReLU
        2:  conv1_2
        3:  relu1_2
        4:  pool1
        5:  conv2_1
        6:  relu2_1   ← We extract AFTER ReLU
        7:  conv2_2
        8:  relu2_2
        9:  pool2
        10: conv3_1
        11: relu3_1   ← We extract AFTER ReLU
        12: conv3_2
        13: relu3_2
        14: conv3_3
        15: relu3_3
        16: conv3_4
        17: relu3_4
        18: pool3
        19: conv4_1
        20: relu4_1   ← We extract AFTER ReLU
        21: conv4_2
        22: relu4_2   ← We extract AFTER ReLU (content layer!)
        23: conv4_3
        24: relu4_3
        25: conv4_4
        26: relu4_4
        27: pool4
        28: conv5_1
        29: relu5_1   ← We extract AFTER ReLU
        30: conv5_2
        31: relu5_2
        32: conv5_3
        33: relu5_3
        34: conv5_4
        35: relu5_4
        """
        # Map layer names to their ReLU indices (what style transfer papers use)
        mapping = {
            'conv1_1': 1,   # relu1_1
            'conv1_2': 3,   # relu1_2
            'conv2_1': 6,   # relu2_1
            'conv2_2': 8,   # relu2_2
            'conv3_1': 11,  # relu3_1
            'conv3_2': 13,  # relu3_2
            'conv3_3': 15,  # relu3_3
            'conv3_4': 17,  # relu3_4
            'conv4_1': 20,  # relu4_1
            'conv4_2': 22,  # relu4_2 ← Content layer
            'conv4_3': 24,  # relu4_3
            'conv4_4': 26,  # relu4_4
            'conv5_1': 29,  # relu5_1
            'conv5_2': 31,  # relu5_2
            'conv5_3': 33,  # relu5_3
            'conv5_4': 35,  # relu5_4
        }
        
        return mapping
    
    def extract_features(self, image_tensor):
        """
        Extract feature maps from the specified layers.
        
        Args:
            image_tensor (torch.Tensor): Input image tensor of shape (N, C, H, W)
                                        normalized for VGG19
        
        Returns:
            dict: Dictionary mapping layer names to feature map tensors.
                  Features are extracted AFTER ReLU activation.
                  IMPORTANT: Gradients are preserved for backpropagation!
        """
        # Dictionary to store features
        features = {}
        
        # Run through all layers sequentially
        x = image_tensor
        
        # Get total number of layers
        num_layers = len(self.model.features)
        
        # Iterate through all layers
        for layer_idx in range(num_layers):
            # Apply the layer
            x = self.model.features[layer_idx](x)
            # If this layer index is one we want to capture
            if layer_idx in self.idx_to_name:
                features[self.idx_to_name[layer_idx]] = x
        
        # Verify all requested layers were captured
        for layer_name in self.layers:
            if layer_name not in features:
                raise RuntimeError(f"Layer '{layer_name}' was not captured. Check layer indices.")
        
        return features
    
    def get_content_features(self, image_tensor):
        """Extract only content layer features."""
        features = self.extract_features(image_tensor)
        return {self.content_layer: features[self.content_layer]}
    
    def get_style_features(self, image_tensor):
        """Extract only style layer features."""
        features = self.extract_features(image_tensor)
        return {layer: features[layer] for layer in self.style_layers}
    
    def __call__(self, image_tensor):
        """Allow calling the instance directly."""
        return self.extract_features(image_tensor)
    

if __name__ == "__main__":
    extractor = VGG19FeatureExtractor()
    img = torch.randn(1,3,224,224, requires_grad=True)
    feats = extractor(img)
    loss = feats['conv4_2'].mean()
    loss.backward()
    print("grad ok:", img.grad is not None and img.grad.abs().sum() > 0)