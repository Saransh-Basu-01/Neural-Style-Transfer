
import torch
import torch.optim as optim
from typing import List, Dict, Tuple, Optional, Union
from collections import OrderedDict
import copy
from src.model.vgg import VGG19FeatureExtractor
from src.model.losses import style_loss,content_loss, gram_matrix

class StyleTransferEngine:
    """
    Core optimization engine for neural style transfer.
    
    This class handles the iterative optimization of pixels to minimize
    content and style losses using a pretrained VGG feature extractor.
    
    Key design decisions:
    - Optimizes pixels (not model parameters)
    - Precomputes target features and Gram matrices once
    - Uses Adam optimizer for stability
    - Clamps generated image to valid range after each step
    """
    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        device: torch.device,
        content_weight: float = 1.0,
        style_weight: float = 1e6,
        content_layer: str = 'conv4_2',
        style_layers: List[str] = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1'],
    ):
        """
        Initialize the style transfer engine.
        
        Args:
            feature_extractor: Pretrained VGG model (with feature extraction method)
            device: Device to run computations on (CPU or CUDA)
            content_weight: Weight for content loss (alpha)
            style_weight: Weight for style loss (beta)
            content_layer: Layer to use for content loss
            style_layers: List of layers to use for style loss
        """
        self.device = device
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_layer = content_layer
        self.style_layers = style_layers
        
        # Move feature extractor to device and set to eval mode
        self.extractor = feature_extractor
        self.extractor.model = self.extractor.model.to(device)
        self.extractor.model.eval()
    
        # Disable gradient for feature extractor parameters
        # This prevents accidentally optimizing VGG weights
        for param in self.extractor.model.parameters():
            param.requires_grad = False
        # Storage for precomputed targets
        self.content_target = None
        self.style_targets = None
        self.content_features = None  # For debugging/visualization
        
        # Loss function instances (will be created during preparation)
        self.content_loss_fn = None
        self.style_loss_fn = None

    def prepare_targets(
        self, 
        content_tensor: torch.Tensor, 
        style_tensor: torch.Tensor
    ) -> None:
        """
        Precompute content target features and style target Gram matrices.
        Called once before optimization to save compute.
        """
        content_tensor = content_tensor.to(self.device)
        style_tensor = style_tensor.to(self.device)
        
        with torch.no_grad():
            # Extract features from content
            content_features = self.extractor.extract_features(content_tensor)
            self.content_target = content_features[self.content_layer].clone()
            
            # Extract features from style
            style_features = self.extractor.extract_features(style_tensor)
            
            # Compute Gram matrices for each style layer
            self.style_targets = {}
            for layer in self.style_layers:
                self.style_targets[layer] = gram_matrix(style_features[layer]).clone().detach()

    # def run(self, content_tensor, num_steps=200, lr=0.5):
    #     if self.content_target is None or self.style_targets is None:
    #         raise ValueError("Call prepare_targets() first")

    #     content_tensor = content_tensor.to(self.device)
    #     generated = content_tensor.clone().detach().requires_grad_(True)

    #     # No line_search_fn, max_iter 10 is more stable than 20
    #     optimizer = optim.LBFGS([generated], lr=lr, max_iter=10)
    #     loss_history = []

    #     for step in range(num_steps):
    #         def closure():
    #             optimizer.zero_grad()
    #             features = self.extractor.extract_features(generated)
    #             content_loss_val = content_loss(self.content_target, features[self.content_layer])
    #             style_loss_val = style_loss(self.style_targets, features)
    #             total_loss = (self.content_weight * content_loss_val +
    #                         self.style_weight * style_loss_val)
    #             total_loss.backward()
    #             return total_loss

    #         loss = optimizer.step(closure)

    #         # fix nan check
    #         if torch.isnan(loss) or torch.isinf(loss):
    #             print(f"NaN detected at step {step}, stopping and returning last good")
    #             break

    #         with torch.no_grad():
    #             generated.data.clamp_(-2.5, 2.8)

    #         loss_history.append(loss.item())

    #         if (step + 1) % 10 == 0:
    #             print(f"Step {step+1}/{num_steps} loss {loss.item():.2f}")

    #     return generated.detach(), loss_history
    def run(self, content_tensor, num_steps=200, lr=0.3):
        if self.content_target is None or self.style_targets is None:
            raise ValueError("Call prepare_targets() first")

        content_tensor = content_tensor.to(self.device)
        generated = content_tensor.clone().detach().requires_grad_(True)

        # history_size 10 and max_iter 5 = very stable
        optimizer = torch.optim.LBFGS([generated], lr=lr, max_iter=5, history_size=10)
        loss_history = []

        for step in range(num_steps):
            def closure():
                optimizer.zero_grad()
                features = self.extractor.extract_features(generated)
                content_loss_val = content_loss(self.content_target, features[self.content_layer])
                style_loss_val = style_loss(self.style_targets, features)
                total_loss = (self.content_weight * content_loss_val +
                            self.style_weight * style_loss_val)
                total_loss.backward()
                # clip insane cubic gradients
                if generated.grad is not None:
                    generated.grad.data.clamp_(-1.0, 1.0)
                return total_loss

            loss = optimizer.step(closure)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN at {step}, returning last good")
                break

            # do not clamp with in-place that breaks LBFGS, use.data
            with torch.no_grad():
                generated.data.clamp_(-2.2, 2.2)

            loss_history.append(loss.item())
            if (step + 1) % 10 == 0:
                print(f"Step {step+1}/{num_steps} loss {loss.item():.2f}")

        return generated.detach(), loss_history

                
