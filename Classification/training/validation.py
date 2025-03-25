import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import numpy as np

def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    top2_correct = 0  # Count of samples where true label is in top-2 predictions
    second_pred_true = 0  # Count of samples where true label is the second prediction

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, mode='classify')
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # Get top-2 predictions
            _, top2_preds = torch.topk(outputs, 2, dim=1)  # Shape: (batch_size, 2)
            predicted = top2_preds[:, 0]  # Top-1 prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Check top-2 accuracy and second prediction
            for i in range(labels.size(0)):
                if labels[i] in top2_preds[i]:
                    top2_correct += 1
                # Check if true label is the second prediction but not the first
                if (labels[i] == top2_preds[i, 1] and labels[i] != top2_preds[i, 0]):
                    second_pred_true += 1
                    
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    top2_accuracy = 100 * top2_correct / total
    second_pred_ratio = 100 * second_pred_true / total if total > 0 else 0.0

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"Validation Results:")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy (Top-1): {val_accuracy:.2f}%")
    print(f"Val Accuracy (Top-2): {top2_accuracy:.2f}%")
    print(f"Ratio of True Label as Second Prediction: {second_pred_ratio:.2f}%")
    print(f"Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}")

    
    return val_loss, val_accuracy, precision, recall, f1, all_preds, all_labels
