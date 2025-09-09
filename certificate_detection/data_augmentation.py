import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
import shutil
from pathlib import Path

class ImageAugmentor:
    def __init__(self, input_dir, output_dir):
        """
        Initialize the augmentor
        Args:
            input_dir: Directory containing original images
            output_dir: Directory to save augmented images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def rotate_image(self, image, angle):
        """Rotate image by given angle"""
        return image.rotate(angle, expand=True, fillcolor='white')
    
    def flip_horizontal(self, image):
        """Flip image horizontally"""
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    
    def flip_vertical(self, image):
        """Flip image vertically"""
        return image.transpose(Image.FLIP_TOP_BOTTOM)
    
    def adjust_brightness(self, image, factor):
        """Adjust brightness (factor: 0.5-2.0)"""
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    
    def adjust_contrast(self, image, factor):
        """Adjust contrast (factor: 0.5-2.0)"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    
    def adjust_saturation(self, image, factor):
        """Adjust saturation (factor: 0.5-2.0)"""
        enhancer = ImageEnhance.Color(image)
        return enhancer.enhance(factor)
    
    def add_noise(self, image):
        """Add random noise to image"""
        img_array = np.array(image)
        noise = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
        noisy_img = cv2.add(img_array, noise)
        return Image.fromarray(noisy_img)
    
    def blur_image(self, image):
        """Apply slight blur"""
        return image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    def sharpen_image(self, image):
        """Sharpen the image"""
        return image.filter(ImageFilter.UnsharpMask())
    
    def crop_and_resize(self, image, crop_factor=0.9):
        """Crop image slightly and resize back to original size"""
        w, h = image.size
        crop_w, crop_h = int(w * crop_factor), int(h * crop_factor)
        
        # Random crop position
        left = random.randint(0, w - crop_w)
        top = random.randint(0, h - crop_h)
        
        cropped = image.crop((left, top, left + crop_w, top + crop_h))
        return cropped.resize((w, h), Image.Resampling.LANCZOS)
    
    def augment_single_image(self, image_path, num_augmentations=10):
        """
        Create multiple augmented versions of a single image
        """
        image = Image.open(image_path).convert('RGB')
        base_name = image_path.stem
        
        # Copy original image
        original_output = self.output_dir / f"{base_name}_original{image_path.suffix}"
        image.save(original_output)
        
        augmented_images = []
        
        for i in range(num_augmentations):
            aug_image = image.copy()
            aug_name = f"{base_name}_aug_{i:02d}{image_path.suffix}"
            
            # Randomly apply 1-3 augmentations
            num_augs = random.randint(1, 3)
            applied_augs = []
            
            for _ in range(num_augs):
                aug_type = random.choice([
                    'rotate', 'brightness', 'contrast', 'saturation',
                    'flip_h', 'flip_v', 'noise', 'blur', 'sharpen', 'crop'
                ])
                
                if aug_type == 'rotate' and 'rotate' not in applied_augs:
                    angle = random.choice([-15, -10, -5, 5, 10, 15])
                    aug_image = self.rotate_image(aug_image, angle)
                    applied_augs.append('rotate')
                    
                elif aug_type == 'brightness' and 'brightness' not in applied_augs:
                    factor = random.uniform(0.7, 1.3)
                    aug_image = self.adjust_brightness(aug_image, factor)
                    applied_augs.append('brightness')
                    
                elif aug_type == 'contrast' and 'contrast' not in applied_augs:
                    factor = random.uniform(0.8, 1.2)
                    aug_image = self.adjust_contrast(aug_image, factor)
                    applied_augs.append('contrast')
                    
                elif aug_type == 'saturation' and 'saturation' not in applied_augs:
                    factor = random.uniform(0.8, 1.2)
                    aug_image = self.adjust_saturation(aug_image, factor)
                    applied_augs.append('saturation')
                    
                elif aug_type == 'flip_h' and 'flip_h' not in applied_augs:
                    aug_image = self.flip_horizontal(aug_image)
                    applied_augs.append('flip_h')
                    
                elif aug_type == 'flip_v' and 'flip_v' not in applied_augs:
                    aug_image = self.flip_vertical(aug_image)
                    applied_augs.append('flip_v')
                    
                elif aug_type == 'noise' and 'noise' not in applied_augs:
                    aug_image = self.add_noise(aug_image)
                    applied_augs.append('noise')
                    
                elif aug_type == 'blur' and 'blur' not in applied_augs:
                    aug_image = self.blur_image(aug_image)
                    applied_augs.append('blur')
                    
                elif aug_type == 'sharpen' and 'sharpen' not in applied_augs:
                    aug_image = self.sharpen_image(aug_image)
                    applied_augs.append('sharpen')
                    
                elif aug_type == 'crop' and 'crop' not in applied_augs:
                    aug_image = self.crop_and_resize(aug_image)
                    applied_augs.append('crop')
            
            # Save augmented image
            output_path = self.output_dir / aug_name
            aug_image.save(output_path)
            augmented_images.append(output_path)
            
        return augmented_images
    
    def augment_dataset(self, augmentations_per_image=15):
        """
        Augment entire dataset
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.input_dir.glob(f"*{ext}"))
            image_files.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No images found in {self.input_dir}")
            return
        
        print(f"Found {len(image_files)} original images")
        print(f"Generating {augmentations_per_image} augmentations per image...")
        
        total_generated = 0
        
        for img_path in image_files:
            try:
                augmented = self.augment_single_image(img_path, augmentations_per_image)
                total_generated += len(augmented)
                print(f"✓ Processed {img_path.name} -> {len(augmented)} new images")
            except Exception as e:
                print(f"✗ Error processing {img_path.name}: {e}")
        
        total_images = len(image_files) + total_generated
        print(f"\n🎉 Augmentation complete!")
        print(f"Original images: {len(image_files)}")
        print(f"Generated images: {total_generated}")
        print(f"Total dataset size: {total_images}")

# Usage Example
if __name__ == "__main__":
    # Set your paths
    input_directory = "Dataset"      # Folder with your 20 original images
    output_directory = "augmented_dataset"   # Folder for all images (original + augmented)
    
    # Create augmentor
    augmentor = ImageAugmentor(input_directory, output_directory)
    
    # Generate 9 augmentations per image (20 * 9 = 180 augmented + 20 original = 200 total)
    augmentor.augment_dataset(augmentations_per_image=9)
    
    print(f"\n🎯 Perfect for labeling!")
    print(f"Your {output_directory} folder now contains ~200 images ready for labeling")
    print(f"Next step: Use makesense.ai or Roboflow to label all images")