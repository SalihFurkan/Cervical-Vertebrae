import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import os
import numpy as np

from data.dataset import XRayDataset, XRayDatasetWithMasks
from data.preprocessor import XRayPreprocessor
from models.attention_efficientnet import AttentionEfficientNet
from models.refinement_classifier import RefinementClassifier
from training.pretraining import pretrain_model, contrastive_pretrain
from training.training import train_model
from training.validation import validate_model
from postprocess.postprocess_refine import postprocess, postprocess_test
from postprocess.confidence_calibration import calibrate_confidence, test_calibration
from evaluation.testing import test_model, test_ensemble_model
from evaluation.gradcam import process_test_dataset_with_gradcam
from utils.loss import CombinedLoss

# Define constants
TARGET_SIZE = 256
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = 8
NUM_CLASSES = 6
LEARNING_RATE = 1e-3
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 16
SEED = 42
SIGMA = 1.0
PRETRAINED_MODEL_PATH = "pretrained_model.pth"
PRETRAIN = False
REFINEMENT = False
CONFIDENCE = True  # Set to True since we’re focusing on confidence calibration
TRAIN = False
K_FOLDS = 5  # Number of folds for cross-validation

# Set random seed for reproducibility
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# Define device globally
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess dataset
def load_dataset(data_dir, mask_dir="dataset/masks", train=True):
    preprocessor = XRayPreprocessor(train=train, target_size=TARGET_SIZE)
    image_paths = []
    labels = []
    mask_paths = []
    class_to_label = {'CS1': 0, 'CS2': 1, 'CS3': 2, 'CS4': 3, 'CS5': 4, 'CS6': 5}

    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path) and class_folder in class_to_label:
            label = class_to_label[class_folder]
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                image_paths.append(image_path)
                labels.append(label)
                mask_filename = image_file.replace(os.path.splitext(image_file)[1], "_mask.png")
                mask_path = os.path.join(mask_dir, mask_filename)
                mask_paths.append(mask_path if os.path.exists(mask_path) else None)

    dataset = XRayDataset(image_paths, labels if train else None, transform=preprocessor)
    return dataset, image_paths, labels, mask_paths

# Generate soft targets for ordinal classes
def generate_soft_targets(num_classes, sigma=1.0):
    soft_targets = torch.zeros(num_classes, num_classes)
    for k in range(num_classes):
        for i in range(num_classes):
            soft_targets[k, i] = torch.exp(torch.tensor(-((i - k) ** 2) / (2 * sigma ** 2)))
        soft_targets[k] /= soft_targets[k].sum()
    return soft_targets

if __name__ == "__main__":
    data_dir = "dataset"
    mask_dir = "dataset/masks"
    train_dataset_full, all_image_paths, all_labels, mask_paths = load_dataset(data_dir, mask_dir, train=True)

    # Separate images with and without masks
    images_with_masks = []
    labels_with_masks = []
    images_without_masks = []
    labels_without_masks = []
    mask_paths_with_masks = []

    for idx, (image_path, label, mask_path) in enumerate(zip(all_image_paths, all_labels, mask_paths)):
        if mask_path is not None:
            images_with_masks.append(image_path)
            labels_with_masks.append(label)
            mask_paths_with_masks.append(mask_path)
        else:
            images_without_masks.append(image_path)
            labels_without_masks.append(label)

    # Ensure images with masks are in the training set
    indices_with_masks = [all_image_paths.index(img) for img in images_with_masks]

    # Split images without masks into train+val and test
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.7, test_size=0.3, random_state=SEED)
    remaining_indices = [i for i in range(len(all_image_paths)) if i not in indices_with_masks]
    train_val_idx_without_masks, test_idx = next(sss.split(np.zeros(len(remaining_indices)), [all_labels[i] for i in remaining_indices]))

    # Map indices back to the original dataset
    train_val_idx = indices_with_masks + [remaining_indices[i] for i in train_val_idx_without_masks]
    test_idx = [remaining_indices[i] for i in test_idx]

    # Create train+val dataset (for k-fold) and test dataset
    train_val_dataset = Subset(train_dataset_full, train_val_idx)
    train_val_labels = [all_labels[i] for i in train_val_idx]
    test_image_paths = [all_image_paths[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]
    test_dataset = XRayDataset(test_image_paths, test_labels, transform=XRayPreprocessor(train=False, target_size=TARGET_SIZE), augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # Prepare pretrain dataset (using all train+val data)
    pretrain_image_paths = [train_dataset_full.image_paths[i] for i in train_val_idx]
    pretrain_dataset = XRayDataset(pretrain_image_paths, labels=None, transform=XRayPreprocessor(train=False, target_size=TARGET_SIZE), augment=False)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Create ROI dataset for images with masks
    roi_labels = labels_with_masks
    roi_dataset = XRayDatasetWithMasks(images_with_masks, mask_paths_with_masks, roi_labels, train_val_idx, all_image_paths, transform=XRayPreprocessor(train=False, target_size=TARGET_SIZE), augment=True)
    roi_loader = DataLoader(roi_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Setup k-fold cross-validation
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
    fold_results = []

    for fold, (train_idx_fold, val_idx_fold) in enumerate(skf.split(np.zeros(len(train_val_idx)), train_val_labels)):
        print(f"\nFold {fold + 1}/{K_FOLDS}")

        # Map fold indices back to the original dataset
        train_idx = [train_val_idx[i] for i in train_idx_fold]
        val_idx = [train_val_idx[i] for i in val_idx_fold]

        # Create fold-specific datasets
        train_dataset = Subset(train_dataset_full, train_idx)
        val_image_paths = [all_image_paths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        val_dataset = XRayDataset(val_image_paths, val_labels, transform=XRayPreprocessor(train=False, target_size=TARGET_SIZE), augment=False)

        # Create data loaders for this fold
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        # Calculate class weights for this fold
        train_labels = [train_dataset_full.labels[idx] for idx in train_idx]
        class_counts = Counter(train_labels)
        total_samples = len(train_labels)
        class_weights = {cls: total_samples / class_counts[cls] for cls in range(NUM_CLASSES)}
        min_weight = min(class_weights.values())
        class_weights = {cls: weight / min_weight for cls, weight in class_weights.items()}
        weights = torch.tensor([class_weights[i] for i in range(NUM_CLASSES)], dtype=torch.float32).to(device)

        # Generate soft targets
        soft_targets = generate_soft_targets(NUM_CLASSES, SIGMA).to(device)

        # Initialize model for this fold
        model = AttentionEfficientNet(device, num_classes=NUM_CLASSES).to(device)

        # Pretraining setup
        pretrain_criterion = nn.MSELoss()
        pretrain_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        if not PRETRAIN and os.path.exists(PRETRAINED_MODEL_PATH):
            print(f"Loading pretrained model from {PRETRAINED_MODEL_PATH}...")
            model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH))
            print("Pretrained model loaded successfully!")
        else:
            print("Starting self-supervised pretraining... (Regular Pretraining)")
            model = pretrain_model(model, pretrain_loader, pretrain_criterion, pretrain_optimizer, num_epochs=10, device=device)
            print("Pretraining completed.")

        # Freeze first half of feature layers
        num_layers = len(model.features)
        for param in model.features[:num_layers//2].parameters():
            param.requires_grad = False

        # Fine-tuning setup
        alpha = 0.0
        beta = 1.0
        gamma = 1.0
        criterion = CombinedLoss(NUM_CLASSES, alpha, beta, gamma, class_weights=weights, soft_targets=soft_targets)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)

        # Train the model for this fold
        if TRAIN:
            model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS, PATIENCE, device, fold)
        else:
            model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))

        # Test the model on the validation fold (optional, for fold-wise performance)
        val_loss, val_accuracy, val_top2_accuracy, val_precision, val_recall, val_f1, val_second_pred_ratio = test_model(model, val_loader, criterion, device)
        print(f"\nFold {fold + 1} Validation Results:")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val Top2 Accuracy: {val_top2_accuracy:.2f}%, Val Second Pred Ratio: {val_second_pred_ratio:.2f}%")
        print(f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}")

        # Confidence calibration for this fold
        if CONFIDENCE:
            platt, iso_reg = calibrate_confidence(model, val_loader, device)
            # Store the calibrator for later use on the test set
            fold_results.append({
                'model_path': f"best_model_fold_{fold}.pth",
                'platt': platt,
                'iso_reg': iso_reg,
                'val_accuracy': val_accuracy,
                'val_top2_accuracy': val_top2_accuracy,
                'val_f1': val_f1
            })

    # Aggregate results across folds
    avg_val_accuracy = np.mean([result['val_accuracy'] for result in fold_results])
    avg_val_top2_accuracy = np.mean([result['val_top2_accuracy'] for result in fold_results])
    avg_val_f1 = np.mean([result['val_f1'] for result in fold_results])
    print(f"\nCross-Validation Results (Average over {K_FOLDS} folds):")
    print(f"Average Validation Accuracy: {avg_val_accuracy:.2f}%")
    print(f"Average Validation Top-2 Accuracy: {avg_val_top2_accuracy:.2f}%")
    print(f"Average Validation F1: {avg_val_f1:.4f}")

    # Select the best fold based on validation accuracy
    best_fold = max(fold_results, key=lambda x: x['val_accuracy'])
    print(f"\nBest fold: {fold_results.index(best_fold) + 1} with Val Accuracy: {best_fold['val_accuracy']:.2f}%")

    # Ensemble: Load all models and combine predictions
    print("\nCreating ensemble from k-fold models...")
    models = []
    for fold in range(K_FOLDS):
        model = AttentionEfficientNet(device, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(f"best_model_fold_{fold}.pth"))
        model.eval()
        models.append(model)

    # Ensemble prediction on test set
    criterion = CombinedLoss(NUM_CLASSES, alpha, beta, gamma, class_weights=weights, soft_targets=soft_targets)
    test_loss, test_accuracy, test_top2_accuracy, test_precision, test_recall, test_f1, test_second_pred_ratio = test_ensemble_model(models, fold_results, test_loader, criterion, CONFIDENCE, device)

    # Optional: GradCAM on test set (using the first model as a representative)
    # process_test_dataset_with_gradcam(models[0], test_dataset, train_dataset_full, device)
