import os
import sys
import numpy as np
from PIL import Image

# Define a simplified img_to_array function to replace keras.preprocessing.image.img_to_array
def img_to_array(img):
    """Converts a PIL Image to a numpy array."""
    x = np.asarray(img, dtype=np.float32)
    if len(x.shape) == 2:
        x = np.expand_dims(x, axis=2)  # Add channel dimension for grayscale
    return x

# Define a simplified version of preprocess_image that doesn't rely on Keras
def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess an image for model prediction without requiring Keras"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Basic normalization as a fallback
        img_array = img_array / 255.0
            
        return img_array, None
    except Exception as e:
        return None, str(e)

# Test the function with a specific image
def test_with_image():
    # Use the project header image
    image_path = "deepfake/images/assets/project_header_image.jpg"
    if os.path.exists(image_path):
        print(f"Testing with image: {image_path}")
        
        # Process the image
        img_array, error = preprocess_image(image_path)
        
        if error:
            print(f"Error processing image: {error}")
        else:
            print(f"Image processed successfully!")
            print(f"Array shape: {img_array.shape}")
            print(f"Array min: {np.min(img_array)}, max: {np.max(img_array)}")
    else:
        print(f"Image file {image_path} not found.")

if __name__ == "__main__":
    test_with_image() 