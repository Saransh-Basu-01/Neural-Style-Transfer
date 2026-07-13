import torch

# 1. Should return True
print("CUDA Available:", torch.cuda.is_available())

# 2. Should print 'GeForce RTX 4060'
if torch.cuda.is_available():
    print("GPU Device Name:", torch.cuda.get_device_name(0))
