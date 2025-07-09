"""
Quick Demo Script for Bilingual Image Captioning

This script provides a simple demonstration of the bilingual image captioning
functionality using a lightweight model for quick testing.
"""

import os
import time
from PIL import Image

from model import BLIPCaptioner, VietnameseTranslator
from model.utils import setup_logging

# Setup logging
setup_logging(log_file="demo.log", level="INFO")

def demo_single_image(image_path: str, model_name: str = "Salesforce/blip-image-captioning-base"):
    """
    Demonstrate captioning on a single image
    
    Args:
        image_path: Path to the image file
        model_name: Model to use for captioning (default: BLIP Base for speed)
    """
    
    print(f"\nDemonstrating bilingual captioning on: {os.path.basename(image_path)}")
    print("-" * 60)
    
    try:
        # Load image
        image = Image.open(image_path)
        print(f"Image loaded: {image.width}x{image.height} pixels")
        
        # Initialize models
        print("Loading captioning model...")
        captioner = BLIPCaptioner(model_name)
        
        print("Loading Vietnamese translator...")
        translator = VietnameseTranslator(primary_service="google")
        
        # Generate English caption
        print("Generating English caption...")
        start_time = time.time()
        english_caption = captioner.generate_caption(
            image=image,
            max_length=50,
            num_beams=3
        )
        generation_time = time.time() - start_time
        
        # Translate to Vietnamese
        print("Translating to Vietnamese...")
        start_time = time.time()
        vietnamese_caption = translator.translate(english_caption)
        translation_time = time.time() - start_time
        
        # Display results
        print("\nRESULTS:")
        print(f"English:    {english_caption}")
        print(f"Vietnamese: {vietnamese_caption}")
        print(f"\nTiming:")
        print(f"Generation: {generation_time:.2f}s")
        print(f"Translation: {translation_time:.2f}s")
        print(f"Total: {generation_time + translation_time:.2f}s")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def demo_multiple_images(image_dir: str = "data", max_images: int = 3):
    """
    Demonstrate captioning on multiple sample images
    
    Args:
        image_dir: Directory containing images
        max_images: Maximum number of images to process
    """
    
    print("\n" + "="*80)
    print("BILINGUAL IMAGE CAPTIONING DEMO")
    print("="*80)
    
    # Find image files
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = []
    
    if os.path.exists(image_dir):
        for file in os.listdir(image_dir):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(image_dir, file))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return False
    
    # Limit number of images
    image_files = image_files[:max_images]
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    successful = 0
    total_time = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}]", end="")
        start_time = time.time()
        
        success = demo_single_image(image_path)
        
        elapsed = time.time() - start_time
        total_time += elapsed
        
        if success:
            successful += 1
        
        print(f"Image processing time: {elapsed:.2f}s")
    
    # Summary
    print("\n" + "="*80)
    print("DEMO SUMMARY")
    print("="*80)
    print(f"Images processed: {successful}/{len(image_files)}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time per image: {total_time/len(image_files):.2f}s")
    
    if successful == len(image_files):
        print("Demo completed successfully!")
    else:
        print(f"Some images failed to process ({len(image_files) - successful} failures)")
    
    return successful > 0


def test_basic_functionality():
    """Test basic import and initialization functionality"""
    
    print("Testing basic functionality...")
    print("-" * 40)
    
    try:
        # Test imports
        print("Testing imports...")
        from model import BLIPCaptioner, VietnameseTranslator
        print("Imports successful")
        
        # Test model initialization (without loading weights)
        print("Testing model initialization...")
        captioner = BLIPCaptioner("Salesforce/blip-image-captioning-base")
        print("BLIP model initialized")
        
        translator = VietnameseTranslator()
        print("Vietnamese translator initialized")
        
        # Test simple translation
        print("Testing translation...")
        test_text = "A cat sitting on a table"
        vietnamese_result = translator.translate(test_text)
        print(f"Translation test: '{test_text}' -> '{vietnamese_result}'")
        
        print("\nBasic functionality test: PASSED")
        return True
        
    except Exception as e:
        print(f"\nBasic functionality test: FAILED - {e}")
        return False


def main():
    """Main demo function"""
    
    print("Starting Bilingual Image Captioning Demo")
    print("Using BLIP Base model for fast demonstration")
    print()
    
    # Test basic functionality first
    if not test_basic_functionality():
        print("Basic functionality test failed. Please check your installation.")
        return
    
    print("\n" + "="*50)
    
    # Check if sample images are available
    if os.path.exists("data") and any(f.endswith(('.jpg', '.jpeg', '.png')) 
                                     for f in os.listdir("data")):
        print("Sample images found. Running multi-image demo...")
        demo_multiple_images()
    else:
        print("No sample images found in 'data' directory.")
        print("Please add some image files to the 'data' directory for testing.")
        
        # Create a simple test image if PIL is available
        try:
            from PIL import Image
            import numpy as np
            
            print("Creating a simple test image...")
            test_image = Image.new('RGB', (100, 100), color='blue')
            test_path = "test_image.jpg"
            test_image.save(test_path)
            
            print(f"Created test image: {test_path}")
            demo_single_image(test_path)
            
            # Clean up
            os.remove(test_path)
            
        except Exception as e:
            print(f"Could not create test image: {e}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main() 