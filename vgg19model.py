import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# Load model
model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

print("=" * 60)
print("VGG19 Features Module - Layer by Layer")
print("=" * 60)

# Print each layer with its index
for i, layer in enumerate(model.features):
    layer_type = layer.__class__.__name__
    if layer_type == 'Conv2d':
        print(f"{i:2d}: Conv2d ({layer.in_channels}->{layer.out_channels})")
    elif layer_type == 'ReLU':
        print(f"{i:2d}: ReLU")
    elif layer_type == 'MaxPool2d':
        print(f"{i:2d}: MaxPool2d")
    else:
        print(f"{i:2d}: {layer_type}")

print(f"\nTotal layers: {len(model.features)}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# Show the mapping for style transfer
print("\n" + "=" * 60)
print("Style Transfer Layer Mapping (after ReLU)")
print("=" * 60)

layer_indices = {
    'conv1_1': 1,
    'conv2_1': 6,
    'conv3_1': 11,
    'conv4_1': 20,
    'conv4_2': 22,
    'conv5_1': 29,
}

for name, idx in layer_indices.items():
    print(f"{name:8s} -> index {idx:2d} ({model.features[idx]})")