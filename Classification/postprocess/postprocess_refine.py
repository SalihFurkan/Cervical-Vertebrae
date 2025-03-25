import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def should_refine(logits, labels):
    
    softmax_probs = torch.softmax(logits, dim=1)
    top2_vals, top2_preds = torch.topk(softmax_probs, k=2, dim=1)
    confidence_ratio = top2_vals[:, 0] / (top2_vals[:, 1] + 1e-6)  # Avoid division by zero
    
    # Check if the label is in the top 2 predictions
    label_in_top2 = torch.sum(top2_preds == labels.unsqueeze(1), dim=1) > 0
    
    # Ratio Condition
    ratio_condition = confidence_ratio < 1.5
    
    uncertain_samples = label_in_top2 & ratio_condition
    
    return uncertain_samples # Refine only if top-2 are close in probability and label is in top2 pred.
    
def calculate_entropy_and_ratio(logits):
  """
  Calculates and concatenates the entropy of the probability distribution
  and the ratio of the best prediction to the second best prediction.

  Args:
    logits (torch.Tensor): Tensor of logits with shape (B, 6), where B is the batch size.

  Returns:
    torch.Tensor: Tensor of shape (B, 2) containing the concatenated
                  entropy and ratio for each batch element.
  """
  # 1. Convert logits to probabilities using softmax
  probabilities = F.softmax(logits, dim=-1)

  # 2. Calculate entropy
  # Handle cases where probability is 0 to avoid log(0) errors
  epsilon = 1e-12
  log_probs = torch.log2(probabilities + epsilon)
  entropy = -torch.sum(probabilities * log_probs, dim=-1, keepdim=True)

  # 3. Find the top two probabilities
  topk_results = torch.topk(probabilities, 2, dim=-1)
  top2_probabilities = topk_results.values

  # 4. Calculate the ratio of the best to the second best probability
  best_probability = top2_probabilities[:, 0].unsqueeze(dim=-1)
  second_best_probability = top2_probabilities[:, 1].unsqueeze(dim=-1)

  # Avoid division by zero if the second best probability is zero (though unlikely after softmax)
  ratio = best_probability / (second_best_probability + epsilon)

  # 5. Concatenate entropy and ratio
  output = torch.cat((probabilities, entropy, ratio), dim=-1)

  return output
  
def create_target_vector_of_size2(logits, true_labels, device):
    """
    Creates a batch of target vectors of size (B, 2) based on the logits and true labels.

    Args:
        logits (torch.Tensor): A torch.Tensor of shape (B, 6) containing the logits for each batch element.
        true_labels (torch.Tensor): A torch.Tensor of shape (B,) containing the true labels for each batch element.

    Returns:
        torch.Tensor or None: A torch.Tensor of shape (B, 2) containing the target vectors,
                               or None if any true label is not in the top 2 predictions.
    """

    if not isinstance(logits, torch.Tensor) or logits.shape[1] != 6:
        raise ValueError("Logits must be a torch.Tensor of shape (B, 6)")

    if not isinstance(true_labels, torch.Tensor) or true_labels.shape != (logits.shape[0],):
        raise ValueError("True labels must be a torch.Tensor of shape (B,)")

    batch_size = logits.shape[0]
    target_vector_one = torch.tensor([1.0, 0.0], dtype=torch.float32).repeat(batch_size,1).to(device)
    target_vector_two = torch.tensor([0.0, 1.0], dtype=torch.float32).repeat(batch_size,1).to(device)
    
    top2_indices = torch.topk(logits, 2, dim=1).indices
    
    is_target_in_first = torch.sum(top2_indices[:,0] == true_labels.unsqueeze(1), dim=1) > 0
    
    target_vector = torch.where(is_target_in_first.unsqueeze(1), target_vector_one, target_vector_two)

    return target_vector


def postprocess(base_model, refinement_model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, threshold, device):
    
    
    best_val_acc = float(0)
    patience_counter = 0
    best_model_path = "best_model_refinement.pth"
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,  eta_min=1e-6)

    for epoch in range(num_epochs):
        base_model.eval()
        refinement_model.train()
                
        total = 0
        total_refine = 0 # Count of samples where the refinement is done
        running_loss = 0.0
        base_correct = 0
        refined_correct = 0 # Count of samples where the refinement improved the result
        top2_correct = 0  # Count of samples where true label is in top-2 predictions
        refined_ = False
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = base_model(images, mode='classify')
            
            # Get top-2 predictions
            top2_vals, top2_preds = torch.topk(logits, 2, dim=1)  # Shape: (batch_size, 2)
            predicted = top2_preds[:, 0]  # Top-1 prediction
            total += labels.size(0)
            base_correct += (predicted == labels).sum().item()
                        
            # Check top-2 accuracy and second prediction
            for i in range(labels.size(0)):
                if labels[i] in top2_preds[i]:
                    top2_correct += 1
            
            uncertain_samples = should_refine(logits, labels)
            refined_preds = top2_preds[:, 0]  # Default to Top-1
            
            if uncertain_samples.any():
                optimizer.zero_grad()
            
                refinement_input = calculate_entropy_and_ratio(logits)
                
                unc_vals_selected = refinement_input[uncertain_samples]
#                true_labels_selected = (top2_preds[:, 1] == labels)[uncertain_samples].to(dtype=torch.float32)
                true_labels_selected = create_target_vector_of_size2(logits, labels, device)[uncertain_samples]
                    
                outputs = refinement_model(unc_vals_selected)
                
#                if sum(uncertain_samples) == 1:
#                    outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, true_labels_selected)  # Compute loss
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                total_refine += uncertain_samples.sum().item()
                
#                switch_to_top2 = outputs > threshold  # Decide switching (only where applicable)
                switch_to_top2 = (outputs[:, 1] > outputs[:, 0])  # Decide switching (only where applicable)
                refined_preds[uncertain_samples] = torch.where(switch_to_top2, top2_preds[uncertain_samples, 1], top2_preds[uncertain_samples, 0])
                if switch_to_top2.any():
                    refined_ = True
                                    
            refined_correct += (refined_preds == labels).float().sum().item()
            
                    
        epoch_loss = running_loss / total_refine
        epoch_refined_accuracy = 100 * refined_correct / total
        epoch_accuracy = 100 * base_correct / total
        epoch_top2_accuracy = 100 * top2_correct / total
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {epoch_loss:.4f}, Train Refined Accuracy (Top-1): {epoch_refined_accuracy:.2f}%")
        print(f"Train Top2 Accuracy (Top-2): {epoch_top2_accuracy:.2f}%,  Train PreRefinement Accuracy: {epoch_accuracy:.2f}%")
        print(f"Train Refined? : {refined_}")
        
        val_loss, val_accuracy, val_refined_accuracy, val_precision, val_recall, val_f1, _, _ = postprocess_validate(base_model, refinement_model, val_loader, criterion, threshold, device)
        
        if val_refined_accuracy > best_val_acc:
            best_val_acc = val_refined_accuracy
            patience_counter = 0
            torch.save(refinement_model.state_dict(), best_model_path)
            print(f"Saved best model with Val Refined Acc: {best_val_acc:.4f}")
            print()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                print()
                break
                
    return refinement_model
    
def postprocess_validate(base_model, refinement_model, val_loader, criterion, threshold, device):
    base_model.eval()
    refinement_model.eval()
    
    val_loss = 0.0
    correct = 0
    total = 0
    correct_refine = 0
    total_refine = 0
    all_preds = []
    all_labels = []
    top2_correct = 0  # Count of samples where true label is in top-2 predictions

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = base_model(images, mode='classify')
            
            # Get top-2 predictions
            top2_vals, top2_preds = torch.topk(logits, 2, dim=1)  # Shape: (batch_size, 2)
            predicted = top2_preds[:, 0]  # Top-1 prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                        
            # Check top-2 accuracy and second prediction
            for i in range(labels.size(0)):
                if labels[i] in top2_preds[i]:
                    top2_correct += 1
            
            uncertain_samples = should_refine(logits, labels)
            refined_ = False
            refined_preds = top2_preds[:, 0]  # Default to Top-1
            
            if uncertain_samples.any():
                refinement_input = calculate_entropy_and_ratio(logits)
                
                unc_vals_selected = refinement_input[uncertain_samples]
#                true_labels_selected = (top2_preds[:, 1] == labels)[uncertain_samples].to(dtype=torch.float32)
                true_labels_selected = create_target_vector_of_size2(logits, labels, device)[uncertain_samples]
                    
                outputs = refinement_model(unc_vals_selected)
                
#                if sum(uncertain_samples) == 1:
#                    outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, true_labels_selected)  # Compute loss
                
                val_loss += loss.item()
                total_refine += uncertain_samples.sum().item()
                                
#                switch_to_top2 = outputs > threshold  # Decide switching (only where applicable)
                switch_to_top2 = (outputs[:, 1] > outputs[:, 0]) # Decide switching (only where applicable)
                refined_preds[uncertain_samples] = torch.where(switch_to_top2, top2_preds[uncertain_samples, 1], top2_preds[uncertain_samples, 0])
                if switch_to_top2.any():
                    refined_ = True
                                    
            correct_refine += (refined_preds == labels).float().sum().item()
            
            all_preds.extend(refined_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
                    
    val_loss = val_loss / total_refine
    val_accuracy = 100 * correct / total
    top2_accuracy = 100 * top2_correct / total
    val_refined_accuracy = 100 * correct_refine / total

    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"Validation Results:")
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy (Top-1): {val_accuracy:.2f}%")
    print(f"Val Refined Accuracy (Top-1 Refined): {val_refined_accuracy:.2f}%")
    print(f"Val Accuracy (Top-2): {top2_accuracy:.2f}%")
    print(f"Val Precision: {precision:.4f}, Val Recall: {recall:.4f}, Val F1: {f1:.4f}")

    
    return val_loss, val_accuracy, val_refined_accuracy, precision, recall, f1, all_preds, all_labels

    

def postprocess_test(base_model, refinement_model, test_loader, criterion, threshold, device):
    base_model.eval()
    refinement_model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    correct_refine = 0
    total_refine = 0
    all_preds = []
    all_labels = []
    top2_correct = 0  # Count of samples where true label is in top-2 predictions
    

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            logits = base_model(images, mode='classify')
            
            # Get top-2 predictions
            top2_vals, top2_preds = torch.topk(logits, 2, dim=1)  # Shape: (batch_size, 2)
            predicted = top2_preds[:, 0]  # Top-1 prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
                        
            # Check top-2 accuracy and second prediction
            for i in range(labels.size(0)):
                if labels[i] in top2_preds[i]:
                    top2_correct += 1
            
            uncertain_samples = should_refine(logits, labels)
            refined_ = False
            refined_preds = top2_preds[:, 0]  # Default to Top-1
            
            if uncertain_samples.any():
                refinement_input = calculate_entropy_and_ratio(logits)
                
                unc_vals_selected = refinement_input[uncertain_samples]
#                true_labels_selected = (top2_preds[:, 1] == labels)[uncertain_samples].to(dtype=torch.float32)
                true_labels_selected = create_target_vector_of_size2(logits, labels, device)[uncertain_samples]
                    
                outputs = refinement_model(unc_vals_selected)
                
#                if sum(uncertain_samples) == 1:
#                    outputs = outputs.unsqueeze(0)
                loss = criterion(outputs, true_labels_selected)  # Compute loss
                
                test_loss += loss.item()
                total_refine += uncertain_samples.sum().item()
                                
#                switch_to_top2 = outputs > threshold  # Decide switching (only where applicable)
                switch_to_top2 = (outputs[:, 1] > outputs[:, 0])  # Decide switching (only where applicable)
                refined_preds[uncertain_samples] = torch.where(switch_to_top2, top2_preds[uncertain_samples, 1], top2_preds[uncertain_samples, 0])
                if switch_to_top2.any():
                    refined_ = True
                                    
            correct_refine += (refined_preds == labels).float().sum().item()
            
            all_preds.extend(refined_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())


    test_loss = test_loss / total_refine
    test_accuracy = 100 * correct / total
    top2_accuracy = 100 * top2_correct / total
    refined_accuracy = 100 * correct_refine / total
    
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[f'CS{i+1}' for i in range(6)], yticklabels=[f'CS{i+1}' for i in range(6)])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('Confusion Matrix After Refinement.png')
    plt.show()
    
    return test_loss, test_accuracy, top2_accuracy, refined_accuracy, precision, recall, f1
