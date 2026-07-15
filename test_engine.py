# test_engine.py
import torch
from src.model.vgg import VGG19FeatureExtractor
from src.model.transfer_engine import StyleTransferEngine
from src.utils.image_processing import load_image, preprocess_vgg, deprocess_vgg
import matplotlib.pyplot as plt

def test_engine():
    """Test the style transfer engine with 10 steps."""
    print("=" * 60)
    print("TESTING STYLE TRANSFER ENGINE")
    print("=" * 60)
    
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n1. Device: {device}")
    
    # 2. Load images (create dummy if needed)
    print("\n2. Loading images...")
    try:
        # Try to load actual images
        content_pil = load_image('content.jpg', max_size=256)
        style_pil = load_image('style.jpg', max_size=256)
        print("   ✓ Loaded content.jpg and style.jpg")
        
        # Preprocess
        print("\n3. Preprocessing images...")
        content_tensor = preprocess_vgg(content_pil).to(device)
        style_tensor = preprocess_vgg(style_pil).to(device)
        print(f"   Content tensor shape: {content_tensor.shape}")
        print(f"   Style tensor shape: {style_tensor.shape}")
        print(f"   Content tensor range: [{content_tensor.min():.2f}, {content_tensor.max():.2f}]")
        
        use_real_images = True
        
    except FileNotFoundError:
        print("   ⚠️  No images found, creating dummy tensors...")
        
        # Create dummy images (random noise)
        dummy_content = torch.randn(1, 3, 224, 224) * 0.5
        dummy_style = torch.randn(1, 3, 224, 224) * 0.5
        
        # Move to device
        content_tensor = dummy_content.to(device)
        style_tensor = dummy_style.to(device)
        print("   ✓ Created dummy tensors")
        print(f"   Content tensor shape: {content_tensor.shape}")
        print(f"   Style tensor shape: {style_tensor.shape}")
        
        use_real_images = False
    
    # 4. Create VGG feature extractor
    print("\n4. Creating VGG feature extractor...")
    vgg = VGG19FeatureExtractor(
        content_layer='conv4_2',
        style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    )
    print(f"   ✓ VGG loaded")
    
    # 5. Create engine
    print("\n5. Creating StyleTransferEngine...")
    engine = StyleTransferEngine(
        feature_extractor=vgg,
        device=device,
        content_weight=1.0,
        style_weight=1e6,
        content_layer='conv4_2',
        style_layers=['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
    )
    print("   ✓ Engine created")
    
    # 6. Prepare targets
    print("\n6. Preparing targets (precomputing features)...")
    engine.prepare_targets(content_tensor, style_tensor)
    print(f"   Content target shape: {engine.content_target.shape}")
    print(f"   Style layers: {list(engine.style_targets.keys())}")
    
    # 7. Run optimization
    print("\n7. Running optimization (10 steps)...")
    print("   " + "-" * 40)
    
    result, loss_history = engine.run(
        content_tensor=content_tensor,
        num_steps=10,
        lr=0.02
    )
    
    print("   " + "-" * 40)
    print("   ✓ Optimization complete!")
    
    # 8. Check results
    print("\n8. Analyzing results...")
    print(f"   Result shape: {result.shape}")
    print(f"   Result range: [{result.min():.2f}, {result.max():.2f}]")
    
    # 9. Check if loss dropped
    print("\n9. Checking loss progression...")
    print(f"   Step 0 loss: {loss_history[0]:.4f}")
    print(f"   Step 9 loss: {loss_history[-1]:.4f}")
    
    if loss_history[-1] < loss_history[0]:
        improvement = ((loss_history[0] - loss_history[-1]) / loss_history[0]) * 100
        print(f"   ✓ Loss DROPPED by {improvement:.2f}% - Engine is working!")
        test_passed = True
    else:
        print(f"   ✗ Loss INCREASED - Check engine implementation")
        test_passed = False
    
    # 10. Show loss progression
    print("\n10. Loss progression:")
    for i, loss in enumerate(loss_history):
        bar_length = int((loss / loss_history[0]) * 30) if loss_history[0] > 0 else 0
        bar = "█" * bar_length
        print(f"   Step {i:2d}: {loss:10.4f} {bar}")
    
    # 11. Save result (only if real images were used)
    if use_real_images:
        print("\n11. Saving result...")
        try:
            output_pil = deprocess_vgg(result)
            output_pil.save('test_output_10steps.jpg')
            print("   ✓ Saved to 'test_output_10steps.jpg'")
        except Exception as e:
            print(f"   ✗ Could not save: {e}")
    else:
        print("\n11. Skipping save (dummy data)")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    if test_passed:
        print("✅ ENGINE IS WORKING CORRECTLY!")
        print("   Loss dropped from step 0 to step 9")
        print("   Ready to move to main.py and configs/default.yaml")
    else:
        print("❌ ENGINE NEEDS DEBUGGING")
        print("   Loss did not decrease")
    
    return result, loss_history, test_passed


if __name__ == "__main__":
    result, loss_history, passed = test_engine()