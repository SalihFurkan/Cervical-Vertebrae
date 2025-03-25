import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def test_model(model, test_loader, criterion, device, fold=-1):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    top2_correct = 0  # Count of samples where true label is in top-2 predictions
    second_pred_true = 0  # Count of samples where true label is the second prediction
    
    if fold == -1:
        figure_savepath = "Confusion Matrix.png"
    else:
        figure_savepath = f"Confusion Matrix {fold} fold.png"

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, mode='classify')
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Get top-2 predictions
            _, top2_preds = torch.topk(outputs, 2, dim=1)  # Shape: (batch_size, 2)
            predicted = top2_preds[:, 0]  # Top-1 prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Check top-2 accuracy
            for i in range(labels.size(0)):
                if labels[i] in top2_preds[i]:
                    top2_correct += 1
                # Check if true label is the second prediction but not the first
                if (labels[i] == top2_preds[i, 1] and labels[i] != top2_preds[i, 0]):
                    second_pred_true += 1


    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    top2_accuracy = 100 * top2_correct / total
    second_pred_ratio = 100 * second_pred_true / total if total > 0 else 0.0
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'CS{i+1}' for i in range(6)], yticklabels=[f'CS{i+1}' for i in range(6)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    plt.savefig(figure_savepath)
#    plt.show()
    return test_loss, test_accuracy, top2_accuracy, precision, recall, f1, second_pred_ratio

def test_ensemble_model(models, fold_results, test_loader, criterion, CONFIDENCE, device):
    # Ensemble prediction on test set
    all_test_probs = []
    all_test_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            batch_probs = []

            # Get predictions from each model
            for fold, model in enumerate(models):
                outputs = model(inputs)  # Shape: (batch_size, NUM_CLASSES)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Raw probabilities

                # Apply Platt Scaling to this model's predictions
                if CONFIDENCE:
                    top1_indices = np.argmax(probs, axis=1)
                    top1_logits = outputs.cpu().numpy()[np.arange(outputs.shape[0]), top1_indices]
                    platt = fold_results[fold]['platt']
                    calibrated_probs = platt.predict_proba(top1_logits.reshape(-1, 1))[:, 1]  # Calibrated top-1 probs

                    # Reconstruct full probability distribution (simplified: scale all class probs by calibrated top-1)
                    calibrated_full_probs = probs.copy()
                    for i in range(len(probs)):
                        max_prob = probs[i, top1_indices[i]]
                        if max_prob > 0:
                            scale = calibrated_probs[i] / max_prob
                            calibrated_full_probs[i] *= scale
                            calibrated_full_probs[i] /= calibrated_full_probs[i].sum()  # Renormalize
                    batch_probs.append(calibrated_full_probs)
                else:
                    batch_probs.append(probs)

            # Average probabilities across models
            batch_probs = np.mean(batch_probs, axis=0)  # Shape: (batch_size, NUM_CLASSES)
            all_test_probs.append(batch_probs)
            all_test_labels.append(labels.cpu().numpy())

    # Concatenate results
    test_probs = np.concatenate(all_test_probs, axis=0)  # Shape: (total_test_samples, NUM_CLASSES)
    test_labels = np.concatenate(all_test_labels, axis=0)  # Shape: (total_test_samples,)

    # Compute ensemble predictions and metrics
    test_predictions = np.argmax(test_probs, axis=1)
    test_accuracy = accuracy_score(test_labels, test_predictions) * 100

    # Top-2 accuracy
    top2_indices = np.argsort(test_probs, axis=1)[:, -2:]
    top2_correct = np.any(top2_indices == test_labels[:, None], axis=1)
    test_top2_accuracy = np.mean(top2_correct) * 100

    # Second prediction ratio (how often the second-highest prediction is correct)
    second_pred_indices = top2_indices[:, 0]  # Second-highest probability class
    second_pred_correct = (second_pred_indices == test_labels)
    second_pred_ratio = np.mean(second_pred_correct) * 100

    # Precision, Recall, F1
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(test_labels, test_predictions, average='weighted', zero_division=0)

    # Compute test loss (approximate, using averaged probabilities)
    test_probs_tensor = torch.tensor(test_probs, dtype=torch.float32).to(device)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long).to(device)
    test_loss = criterion(test_probs_tensor.log(), test_labels_tensor).item()

    print(f"\nEnsemble Test Results:")
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Test Top2 Accuracy: {test_top2_accuracy:.2f}%, Test Second Pred Ratio: {second_pred_ratio:.2f}%")
    print(f"Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}")
        
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'CS{i+1}' for i in range(6)], yticklabels=[f'CS{i+1}' for i in range(6)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    plt.savefig("Confusion Matrix for Ensemble Model")
    plt.show()
        
    return test_loss, test_accuracy, test_top2_accuracy, test_precision, test_recall, test_f1, second_pred_ratio
