# create_test_images.py
from PIL import Image, ImageDraw
import numpy as np

def create_test_images():
    """Create simple test images for style transfer."""
    
    # 1. Create a simple content image (a circle on a square)
    content = Image.new('RGB', (256, 256), color=(100, 150, 200))
    draw = ImageDraw.Draw(content)
    draw.ellipse([50, 50, 206, 206], fill=(255, 200, 100), outline=(0, 0, 0))
    draw.rectangle([100, 100, 156, 156], fill=(200, 50, 50))
    content.save('content.jpg')
    print("✓ Created content.jpg (circle on blue background)")
    
    # 2. Create a simple style image (stripes/pattern)
    style = Image.new('RGB', (256, 256), color=(50, 50, 50))
    draw = ImageDraw.Draw(style)
    
    # Draw colorful stripes
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for i, color in enumerate(colors):
        y_start = i * 51
        y_end = (i + 1) * 51
        draw.rectangle([0, y_start, 256, y_end], fill=color)
    
    # Add some texture dots
    for _ in range(100):
        x = np.random.randint(0, 256)
        y = np.random.randint(0, 256)
        size = np.random.randint(2, 8)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        draw.ellipse([x-size, y-size, x+size, y+size], fill=color)
    
    style.save('style.jpg')
    print("✓ Created style.jpg (colorful stripes with dots)")
    
    print("\n✅ Test images created successfully!")
    print("Now run: uv run python test_engine.py")

if __name__ == "__main__":
    create_test_images()