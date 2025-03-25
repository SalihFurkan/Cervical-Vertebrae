import cv2
from PIL import Image, ImageOps
import torchvision.transforms as transforms
import numpy as np

class XRayPreprocessor:
    def __init__(self, train=True, target_size=256):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.target_size = target_size
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x),  # Handle masks
        ])
        if train:
            self.augment = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.1),
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
            ])
        else:
            self.augment = None

    def enhance_contrast(self, image):
        image_np = np.array(image)
        enhanced_image = self.clahe.apply(image_np)
        return Image.fromarray(enhanced_image)

    def resize_with_padding(self, image):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        if original_width > original_height:
            new_height = int(self.target_size / aspect_ratio)
            resized_image = image.resize((self.target_size, new_height), Image.Resampling.BICUBIC)
        else:
            new_width = int(self.target_size * aspect_ratio)
            resized_image = image.resize((new_width, self.target_size), Image.Resampling.BICUBIC)
        padded_image = ImageOps.pad(resized_image, (self.target_size, self.target_size), color=0, centering=(0.5, 0.5))
        return padded_image.convert('L')

    def __call__(self, image):
        enhanced_image = self.enhance_contrast(image)
        padded_image = self.resize_with_padding(enhanced_image)
        tensor_image = self.tensor_transform(padded_image)
        if self.augment and tensor_image.shape[0] == 3:  # Apply augmentations only to images (not masks)
            tensor_image = self.augment(tensor_image)
        return tensor_image
