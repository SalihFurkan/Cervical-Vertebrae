import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
import cv2  # Added for contour detection

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from data.dataset import XRayDatasetWithMasks
from data.preprocessor import XRayPreprocessor

# Define constants (match with your training setup)
TARGET_SIZE = 256
SEED = 42
DATA_DIR = "dataset"
MASK_DIR = "dataset/masks"

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Load image and mask paths
image_paths = []
mask_paths = []
labels = []
class_to_label = {'CS1': 0, 'CS2': 1, 'CS3': 2, 'CS4': 3, 'CS5': 4, 'CS6': 5}

for class_folder in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_folder)
    if os.path.isdir(class_path) and class_folder in class_to_label:
        label = class_to_label[class_folder]
        for image_file in os.listdir(class_path):
            image_path = os.path.join(class_path, image_file)
            image_paths.append(image_path)
            labels.append(label)
            mask_filename = image_file.replace(os.path.splitext(image_file)[1], "_mask.png")
            mask_path = os.path.join(MASK_DIR, mask_filename)
            mask_paths.append(mask_path if os.path.exists(mask_path) else None)

# Filter out images without masks for visualization
valid_pairs = [(img, mask, lbl) for img, mask, lbl in zip(image_paths, mask_paths, labels) if mask is not None]
image_paths_with_masks = [pair[0] for pair in valid_pairs]
mask_paths_with_masks = [pair[1] for pair in valid_pairs]
labels_with_masks = [pair[2] for pair in valid_pairs]

# Create a dummy train_idx (since we’re visualizing all masked images)
train_idx = list(range(len(image_paths_with_masks)))

# Initialize preprocessor and dataset
preprocessor = XRayPreprocessor(train=False, target_size=TARGET_SIZE)
dataset = XRayDatasetWithMasks(image_paths_with_masks, mask_paths_with_masks, labels_with_masks, train_idx, image_paths_with_masks, transform=preprocessor, augment=True)

# Function to overlay mask on image
def overlay_mask(image, mask, method='highlight'):

    image_np = np.array(image)  # Shape: (256, 256) or (256, 256, 3)
    mask_np = np.array(mask)  # Shape: (256, 256)

    if len(image_np.shape) == 2:
        image_np = np.stack([image_np] * 3, axis=-1)  # Convert grayscale to RGB
    elif image_np.shape[2] == 1:
        image_np = np.repeat(image_np, 3, axis=2)

    if method == 'multiply':
        # Element-wise multiplication
        overlaid = image_np * mask_np[..., np.newaxis]  # Broadcast mask to RGB
    elif method == 'highlight':
        # Highlight mask area with red outline
        overlaid = image_np.copy()
        contours, _ = cv2.findContours((mask_np * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlaid, contours, -1, (255, 0, 0), 2)  # Red outline
    else:
        raise ValueError("Method must be 'multiply' or 'highlight'")

    return overlaid.astype(np.uint8)
    
# Function to create a montage of original, multiplied, and highlighted images
def create_montage(image, mask):
    image_np = (image[0,:,:].squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_np = mask[0,:,:].squeeze().cpu().numpy()

    # Generate the three versions
    original = image_np
    multiplied = overlay_mask(image_np, mask_np, method='multiply')
    highlighted = overlay_mask(image_np, mask_np, method='highlight')

    # Ensure all images are in RGB format
    if len(original.shape) == 2:
        original = np.stack([original] * 3, axis=-1)
    if len(multiplied.shape) == 2:
        multiplied = np.stack([multiplied] * 3, axis=-1)
    if len(highlighted.shape) == 2:
        highlighted = np.stack([highlighted] * 3, axis=-1)

    # Concatenate horizontally
    montage = np.concatenate((original, multiplied, highlighted), axis=1)
    return montage

# Visualize masks
num_samples = len(dataset)
n_cols = 4  # Number of columns in subplot
n_rows = (num_samples + n_cols - 1) // n_cols  # Calculate rows needed

print(num_samples, n_cols, n_rows)

plt.figure(figsize=(15, 5 * n_rows))
for i in range(num_samples):
    image, mask, _ = dataset[i]
    
    montage = create_montage(image, mask)

    # Plot the montage
    plt.subplot(n_rows, n_cols, i + 1)
    plt.imshow(montage)
    plt.title(f"Image {i+1}: Original | Multiplied | Highlighted")
    plt.axis('off')

plt.tight_layout()
plt.show()
