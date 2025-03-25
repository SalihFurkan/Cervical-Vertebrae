import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from training.validation import validate_model

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device, fold=-1):
    best_val_loss = float('inf')
    patience_counter = 0
    if fold >= 0:
        best_model_path = f"best_model_fold_{fold}.pth"
    else:
        best_model_path = "best_model.pth"
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs,  eta_min=1e-6)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        top2_correct = 0  # Count of samples where true label is in top-2 predictions
        second_pred_true = 0  # Count of samples where true label is the second prediction
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, mode='classify')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Get top-2 predictions
            top2_vals, top2_preds = torch.topk(outputs, 2, dim=1)  # Shape: (batch_size, 2)
            predicted = top2_preds[:, 0]  # Top-1 prediction
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            running_loss += loss.item()
                        
            # Check top-2 accuracy and second prediction
            for i in range(labels.size(0)):
                if labels[i] in top2_preds[i]:
                    top2_correct += 1
                # Check if true label is the second prediction but not the first
                if (labels[i] == top2_preds[i, 1] and labels[i] != top2_preds[i, 0]):
                    second_pred_true += 1

                    
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        epoch_top2_accuracy = 100 * top2_correct / total
        epoch_second_pred_ratio = 100 * second_pred_true / total if total > 0 else 0.0
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {epoch_loss:.4f}, Train Accuracy (Top-1): {epoch_accuracy:.2f}%")
        print(f"Train Top2 Accuracy (Top-2): {epoch_top2_accuracy:.2f}%,  Train Second Pred Accuracy: {epoch_second_pred_ratio:.2f}%")
        
        val_loss, val_accuracy, val_precision, val_recall, val_f1, _, _ = validate_model(model, val_loader, criterion, device)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")
            print()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                print()
                break
                
    return model

def train_model_with_roi(model, train_loader, val_loader, soft_targets, criterion, optimizer, num_epochs, patience, train_idx, all_image_paths, device, roi_loader=None):
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = "best_model.pth"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    roi_iterator = iter(roi_loader) if roi_loader else None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            roi_mask = None
            # Apply ROI loss for the subset with masks
            if roi_iterator and i < len(roi_loader):
                try:
                    _, masks, _ = next(roi_iterator)
                    roi_mask = masks.to(device)
                    if roi_mask.size(0) < images.size(0):
                        roi_mask = torch.cat([roi_mask, torch.zeros(images.size(0) - roi_mask.size(0), *roi_mask.shape[1:]).to(device)], dim=0)
                    elif roi_mask.size(0) > images.size(0):
                        roi_mask = roi_mask[:images.size(0)]
                except StopIteration:
                    roi_iterator = iter(roi_loader)
                    _, masks, _ = next(roi_iterator)
                    roi_mask = masks.to(device)
                    if roi_mask.size(0) < images.size(0):
                        roi_mask = torch.cat([roi_mask, torch.zeros(images.size(0) - roi_mask.size(0), *roi_mask.shape[1:]).to(device)], dim=0)
                    elif roi_mask.size(0) > images.size(0):
                        roi_mask = roi_mask[:images.size(0)]
            if roi_mask is not None:
                outputs, roi_loss = model(images, mode='classify', mask_roi=roi_mask)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
                loss = - (soft_targets[labels] * log_probs).sum(dim=1).mean()  # Soft label loss
                loss = loss + 0.1 * roi_loss
            else:
                outputs = model(images, mode='classify')
                log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
                loss = - (soft_targets[labels] * log_probs).sum(dim=1).mean()  # Soft label loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_loss = running_loss / len(train_loader)

        val_loss, val_accuracy, val_precision, val_recall, val_f1, _, _ = validate_model(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {epoch_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with Val Loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience_counter} epochs without improvement.")
                break
    model.load_state_dict(torch.load(best_model_path))
    return model
