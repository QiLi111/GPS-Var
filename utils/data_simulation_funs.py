
import numpy as np
import cv2
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
 
def elastic_transform(mask, alpha, sigma, random_seed=0):
    """Apply elastic deformation for simulating annotation drift with continuous strength."""
    np.random.seed(random_seed)
    shape = mask.shape
 
    dx = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
    dy = ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant") * alpha
 
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
 
    distorted_mask = ndimage.map_coordinates(mask, indices, order=1).reshape(shape)
    return (distorted_mask > 0.5).astype(np.uint8)
 
def apply_morphological_bias(mask, bias_level):
    """Continuously apply dilation (positive bias) or erosion (negative bias)."""
    kernel_size = int(abs(bias_level) * 5) + 1  # Convert bias level to kernel size
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if bias_level > 0:  # Over-segmentation (dilate)
        return cv2.dilate(mask, kernel, iterations=1)
    elif bias_level < 0:  # Under-segmentation (erode)
        return cv2.erode(mask, kernel, iterations=1)
    return mask  # No bias (bias_level = 0)
 
def add_continuous_gaussian_noise(mask, noise_level, random_seed=42):
    """Add continuous-controlled Gaussian noise to the mask."""
    np.random.seed(random_seed)
    noise = np.random.normal(0, noise_level, mask.shape)
    noisy_mask = mask + noise
    return (noisy_mask > 0.5).astype(np.uint8)
 
def generate_continuous_annotation(mask, bias_level=0.0, variance_level=0.0):
    """
    Generate a continuously controlled annotation with:
    - `bias_level` controlling over- or under-segmentation (-1 to 1)
    - `variance_level` controlling annotation inconsistency (0 to 1)
    """
    # Apply bias (systematic shift)
    mask = apply_morphological_bias(mask, bias_level)
 
    # Apply variance (random distortion)
    if variance_level > 0:
        mask = elastic_transform(mask, alpha=variance_level * 500, sigma=variance_level * 2.5)
        # mask = add_continuous_gaussian_noise(mask, noise_level=variance_level * 0.1)
 
    return mask
 
# # Example Usage
# h, w = 200, 200
# mask = np.zeros((h, w), dtype=np.uint8)
# cv2.circle(mask, (100, 100), 50, 1, -1)  # Ground-truth circle
 
# # Generate annotations with different bias & variance levels
# masks = [
#     generate_continuous_annotation(mask, bias_level=-0.5, variance_level=0.0),  # Under-segmentation
#     generate_continuous_annotation(mask, bias_level=0.5, variance_level=0.0),   # Over-segmentation
#     generate_continuous_annotation(mask, bias_level=0.0, variance_level=0.5),   # Moderate noise
#     generate_continuous_annotation(mask, bias_level=0.5, variance_level=0.5)    # Over-segmented + noisy
# ]
 
# titles = ["Under-segmentation", "Over-segmentation", "Noisy", "Over-segmentation + Noise"]
 
# # Plot results
# fig, axes = plt.subplots(1, 4, figsize=(15, 5))
# for ax, img, title in zip(axes, masks, titles):
#     ax.imshow(img, cmap="gray")
#     ax.set_title(title)
#     ax.axis("off")
 
# plt.show()
# plt.savefig('continuous_annotation.png')