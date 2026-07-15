def prepare_targets(self, content_tensor, style_tensor):
    # Step 3.1: Move tensors to GPU
    content_tensor = content_tensor.to(self.device)  # [1,3,224,224] on GPU
    style_tensor = style_tensor.to(self.device)      # [1,3,224,224] on GPU
    
    # Step 3.2: Disable gradient tracking for efficiency
    with torch.no_grad():
        # Step 3.3: Extract features from content (dog)
        content_features = self._extract_features(content_tensor)
        # content_features is a dict:
        # {
        #   'conv1_1': [1, 64, 224, 224],  # Early layer, many spatial details
        #   'conv2_1': [1, 128, 112, 112],
        #   'conv3_1': [1, 256, 56, 56],
        #   'conv4_1': [1, 512, 28, 28],
        #   'conv4_2': [1, 512, 28, 28],  # Content layer
        #   'conv5_1': [1, 512, 14, 14]
        # }
        
        # Step 3.4: Store content target
        self.content_target = content_features['conv4_2'].clone()
        # self.content_target: [1, 512, 28, 28]
        # Contains the "essence" of the dog at conv4_2 level
        
        # Step 3.5: Extract features from style (Starry Night)
        style_features = self._extract_features(style_tensor)
        # Same structure as content_features but for style image
        
        # Step 3.6: Compute Gram matrices for each style layer
        self.style_targets = {}
        for layer in ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']:
            # style_features[layer]: [1, C, H, W]
            # gram_matrix reshapes to [1, C, H*W] and computes correlation
            self.style_targets[layer] = gram_matrix(style_features[layer]).clone().detach()
            # Result: [1, C, C] - captures texture patterns