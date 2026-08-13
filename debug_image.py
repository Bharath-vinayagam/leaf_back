import numpy as np
from PIL import Image
import tensorflow as tf

def analyze_image(image_path):
    """Analyze an image to understand its characteristics"""
    # Load and resize image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    print(f"Image: {image_path}")
    print(f"Shape: {img_array.shape}")
    print(f"Data type: {img_array.dtype}")
    print(f"Value range: {img_array.min():.3f} to {img_array.max():.3f}")
    
    # Convert to HSV
    hsv = tf.image.rgb_to_hsv(img_array)
    h = hsv[:, :, 0]  # Hue
    s = hsv[:, :, 1]  # Saturation
    v = hsv[:, :, 2]  # Value
    
    print(f"\nHSV Analysis:")
    print(f"Hue range: {h.numpy().min():.3f} to {h.numpy().max():.3f}")
    print(f"Saturation range: {s.numpy().min():.3f} to {s.numpy().max():.3f}")
    print(f"Value range: {v.numpy().min():.3f} to {v.numpy().max():.3f}")
    
    # Calculate statistics
    mean_hue = tf.reduce_mean(h)
    mean_saturation = tf.reduce_mean(s)
    mean_value = tf.reduce_mean(v)
    
    print(f"\nMean values:")
    print(f"Mean Hue: {mean_hue:.3f}")
    print(f"Mean Saturation: {mean_saturation:.3f}")
    print(f"Mean Value: {mean_value:.3f}")
    
    # Green color analysis
    green_mask = tf.logical_and(
        tf.greater(h, 0.2),  # Green hue range
        tf.less(h, 0.4)
    )
    green_ratio = tf.reduce_mean(tf.cast(green_mask, tf.float32))
    print(f"Green ratio: {green_ratio:.3f}")
    
    # Texture analysis
    gray = tf.image.rgb_to_grayscale(img_array)
    texture_variance = tf.math.reduce_variance(gray)
    print(f"Texture variance: {texture_variance:.5f}")
    
    # RGB analysis
    r = img_array[:, :, 0]
    g = img_array[:, :, 1]
    b = img_array[:, :, 2]
    
    mean_r = tf.reduce_mean(r)
    mean_g = tf.reduce_mean(g)
    mean_b = tf.reduce_mean(b)
    
    print(f"\nRGB Analysis:")
    print(f"Mean Red: {mean_r:.3f}")
    print(f"Mean Green: {mean_g:.3f}")
    print(f"Mean Blue: {mean_b:.3f}")
    
    # Calculate green dominance in RGB
    green_dominance = mean_g / (mean_r + mean_g + mean_b + 1e-8)
    print(f"Green dominance (RGB): {green_dominance:.3f}")
    
    # Edge analysis
    edges_x = tf.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
    edges_y = tf.abs(gray[:, 1:, :, :] - gray[:, :-1, :, :])
    edge_density = tf.reduce_mean(edges_x) + tf.reduce_mean(edges_y)
    print(f"Edge density: {edge_density:.5f}")
    
    # Coverage analysis
    non_zero_pixels = tf.reduce_sum(tf.cast(tf.greater(gray, 0.1), tf.float32))
    coverage_ratio = non_zero_pixels / (224 * 224)
    print(f"Coverage ratio: {coverage_ratio:.3f}")
    
    return img_array

if __name__ == "__main__":
    # Analyze the test images
    print("=" * 50)
    analyze_image('test_leaf.jpg')
    
    print("\n" + "=" * 50)
    analyze_image('test_non_leaf.jpg')
    
    print("\n" + "=" * 50)
    analyze_image('test_real_leaf.jpg')
