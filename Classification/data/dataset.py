from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torch
import torchvision.transforms as transforms

class XRayDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None, augment=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        if self.augment:
            self.spatial_augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        else:
            self.spatial_augment = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
        # Apply the same spatial augmentations to both image and mask
        if self.augment and self.spatial_augment:
            # Convert tensors back to PIL for augmentation
            image_pil = transforms.ToPILImage()(image)
            image_pil = self.spatial_augment(image_pil)
            image = transforms.ToTensor()(image_pil)
        if self.labels is not None:
            return image, self.labels[idx]
        return image

class XRayDatasetWithMasks(Dataset):
    def __init__(self, image_paths, mask_paths, labels, train_idx, all_image_paths, transform=None, target_size=256, augment=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.train_idx = train_idx
        self.all_image_paths = all_image_paths
        self.transform = transform
        self.target_size = target_size
        self.augment = augment
        # Define spatial augmentations suitable for both images and masks
        if self.augment:
            self.spatial_augment = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        else:
            self.spatial_augment = None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Cycle through train_idx to ensure alignment with training set
        original_idx = self.train_idx[idx % len(self.train_idx)]
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Load image and mask
        image = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        # Apply preprocessing (resize, enhance contrast, etc.)
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Apply the same spatial augmentations to both image and mask
        if self.augment and self.spatial_augment:
            # Convert tensors back to PIL for augmentation
            image_pil = transforms.ToPILImage()(image)
            mask_pil = transforms.ToPILImage()(mask)
            # Seed the random state to ensure the same transformations
            seed = torch.randint(0, 2**32, (1,)).item()
            torch.manual_seed(seed)
            image_pil = self.spatial_augment(image_pil)
            torch.manual_seed(seed)  # Use the same seed for mask
            mask_pil = self.spatial_augment(mask_pil)
            # Convert back to tensors
            image = transforms.ToTensor()(image_pil)
            mask = transforms.ToTensor()(mask_pil)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()  # Threshold to maintain binary nature

        return image, mask, self.labels[idx]
