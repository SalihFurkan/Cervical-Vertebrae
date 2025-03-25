import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Compute ECE on test set with calibrated probabilities
def compute_ece(confidences, true_labels, predictions, n_bins=10):
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_bounds[1:-1])
    accuracies, bin_confidences = [], []
    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_acc = accuracy_score(true_labels[mask], predictions[mask])
            bin_conf = confidences[mask].mean()
        else:
            bin_acc = 0
            bin_conf = bin_bounds[i] + (bin_bounds[i + 1] - bin_bounds[i]) / 2
        accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
    ece = np.mean(np.abs(np.array(bin_confidences) - np.array(accuracies)))
    return ece

def calibrate_confidence(base_model, val_loader, device):

    base_model.eval()

    # Collect predictions from val_loader
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for inputs, true_labels in val_loader:
            inputs, true_labels = inputs.to(device), true_labels.to(device)
            outputs = base_model(inputs)  # Shape: (B, 6)
            all_outputs.append(outputs.cpu())
            all_labels.append(true_labels.cpu())

    # Concatenate all batches
    outputs = torch.cat(all_outputs, dim=0)  # Shape: (total_samples, 6)
    true_labels = torch.cat(all_labels, dim=0)  # Shape: (total_samples,)
    B = outputs.shape[0]

    # Convert to probabilities
    probs = torch.softmax(outputs, dim=1).numpy()  # Shape: (B, 6)
    true_labels = true_labels.numpy()  # Shape: (B,)

    # Top-1 predictions and confidences
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    binary_labels = (predictions == true_labels).astype(int)

    # Step 1: Reliability Diagram and ECE
    n_bins = 10
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_bounds[1:-1])
    accuracies, bin_confidences = [], []
    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_acc = accuracy_score(true_labels[mask], predictions[mask])
            bin_conf = confidences[mask].mean()
        else:
            bin_acc = 0
            bin_conf = bin_bounds[i] + (bin_bounds[i + 1] - bin_bounds[i]) / 2
        accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)

    plt.plot(bin_confidences, accuracies, marker='o', label='Before Calibration')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Confidence')
    plt.ylabel('Observed Accuracy')
    plt.legend()

    ece = np.mean(np.abs(np.array(bin_confidences) - np.array(accuracies)))
    print(f"Before Calibration ECE: {ece:.4f}")

    # Step 2: Platt Scaling
    top1_logits = outputs[np.arange(B), predictions].numpy()  # Logits for top-1
    platt = LogisticRegression(solver='lbfgs')
    platt.fit(top1_logits.reshape(-1, 1), binary_labels)
    calibrated_probs_platt = platt.predict_proba(top1_logits.reshape(-1, 1))[:, 1]

    # Step 3: Isotonic Regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(confidences, binary_labels)
    calibrated_probs_iso = iso_reg.predict(confidences)

    # Evaluate calibrated ECE (Platt example)
    bin_indices_cal = np.digitize(calibrated_probs_platt, bin_bounds[1:-1])
    accuracies_cal = []
    for i in range(n_bins):
        mask = (bin_indices_cal == i)
        if mask.sum() > 0:
            bin_acc = accuracy_score(true_labels[mask], predictions[mask])
        else:
            bin_acc = 0
        accuracies_cal.append(bin_acc)

    ece_cal = np.mean(np.abs(np.array(bin_confidences) - np.array(accuracies_cal)))
    print(f"After Platt Calibration ECE: {ece_cal:.4f}")
        
#    plt.savefig('Reliability.png')
#    plt.show()
    
    return platt, iso_reg
    
def test_calibration(base_model, test_loader, platt, iso_reg, device, method="Platt"):

    all_test_outputs = []
    all_test_labels = []
    all_calibrated_probs = []
    with torch.no_grad():
        for inputs, true_labels in test_loader:  # Use test_loader here
            inputs, true_labels = inputs.to(device), true_labels.to(device)
            outputs = base_model(inputs)  # Shape: (B, 6)
            outputs_np = outputs.cpu().numpy()
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            top1_indices = np.argmax(probs, axis=1)
            top1_logits = outputs_np[np.arange(outputs_np.shape[0]), top1_indices]
            calibrated_probs = platt.predict_proba(top1_logits.reshape(-1, 1))[:, 1]
            
            # Store results
            all_test_outputs.append(outputs.cpu())
            all_test_labels.append(true_labels.cpu())
            all_calibrated_probs.append(calibrated_probs)

    # Concatenate test results
    test_outputs = torch.cat(all_test_outputs, dim=0)  # Shape: (total_test_samples, 6)
    test_labels = torch.cat(all_test_labels, dim=0)  # Shape: (total_test_samples,)
    calibrated_test_probs = np.concatenate(all_calibrated_probs, axis=0)  # Shape: (total_test_samples,)

    # Evaluate test set (optional: accuracy and ECE)
    test_probs = torch.softmax(test_outputs, dim=1).numpy()
    test_predictions = np.argmax(test_probs, axis=1)
    test_accuracy = accuracy_score(test_labels.numpy(), test_predictions)
    print(f"Test Set Top-1 Accuracy: {test_accuracy:.4f}")

    ece_test_raw = compute_ece(np.max(test_probs, axis=1), test_labels.numpy(), test_predictions)
    ece_test_cal = compute_ece(calibrated_test_probs, test_labels.numpy(), test_predictions)
    print(f"Test Set ECE (Before Calibration): {ece_test_raw:.4f}")
    print(f"Test Set ECE (After Platt Calibration): {ece_test_cal:.4f}")

   
    # Reliability Diagram
    confidences = calibrated_test_probs
    test_labels = test_labels.numpy()
    test_probs  = np.argmax(test_probs, axis=1)
    n_bins = 10
    bin_bounds = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_bounds[1:-1])
    accuracies, bin_confidences = [], []
    for i in range(n_bins):
        mask = (bin_indices == i)
        if mask.sum() > 0:
            bin_acc = accuracy_score(test_labels[mask], test_probs[mask])
            bin_conf = confidences[mask].mean()
        else:
            bin_acc = 0
            bin_conf = bin_bounds[i] + (bin_bounds[i + 1] - bin_bounds[i]) / 2
        accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)

    plt.plot(bin_confidences, accuracies, marker='o', label='Before Calibration')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Calibration')
    plt.xlabel('Predicted Confidence')
    plt.ylabel('Observed Accuracy')
    plt.legend()
#    plt.savefig("After_Platt_Calibration_reliability.png")
#    plt.show()
