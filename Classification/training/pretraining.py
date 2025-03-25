import torch
import torch.nn as nn
import torch.optim as optim

def pretrain_model(model, pretrain_loader, criterion, optimizer, num_epochs, device, save_path="pretrained_model.pth"):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images in pretrain_loader:
            images = images.to(device)
            true_images = images
            noisy_images = images + 0.1 * torch.randn_like(images, device=device)
            noisy_images = torch.clamp(noisy_images, 0, 1)
            optimizer.zero_grad()
            outputs = model(noisy_images, mode='reconstruct')
            loss = criterion(outputs, true_images)
            loss.backward()
            optimizer.step()
            batch_size = images.shape[0]
            running_loss += (loss.item() * batch_size)  # Scale by batch size
        epoch_loss = running_loss / len(pretrain_loader)
        print(f"Pretraining Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
    # Save the feature extractor
    print("Saving pretrained feature extractor...")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
    
    return model
    
import torch.nn.functional as F
import torchvision.transforms as transforms

def contrastive_pretrain(model, pretrain_loader, criterion, optimizer, num_epochs, device, temperature=0.07, save_path="pretrained_model.pth"):
    """Contrastive pretraining using SimCLR approach"""
    model.to(device)
    model.train()
    
    # Projection head for contrastive learning
    projection = nn.Sequential(
        nn.Linear(1536, 512),
        nn.ReLU(),
        nn.Linear(512, 128)
    ).to(device)
    proj_optimizer = optim.Adam(projection.parameters(), lr=1e-4)
    
    # Augmentations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
        transforms.ToTensor(),  # Ensure the image is a tensor
    ])
    
    def transform_call(batch):
    
        augmented_views = []
        
        # Iterate over the batch and apply transformations to each image
        for image in batch:
            # Convert tensor to PIL image for augmentations
            pil_image = transforms.ToPILImage()(image)
            
            # Apply the transformation
            view_1 = transform(pil_image)
            view_2 = transform(pil_image)  # Generate a second view
            
            augmented_views.append((view_1, view_2))
        
        # Return a batch of augmented views
        return torch.stack([view[0] for view in augmented_views]), torch.stack([view[1] for view in augmented_views])

    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images in pretrain_loader:
#            images = images.to(device)
            aug1, aug2 = transform_call(images)
#            aug2 = transform(images)

            aug1, aug2 = aug1.to(device), aug2.to(device)
            
            # Extract features
            features1 = model.features(aug1)
            features2 = model.features(aug2)
            features1 = model.avg_pool(features1).view(features1.size(0), -1)
            features2 = model.avg_pool(features2).view(features2.size(0), -1)
            
            # Project to embedding space
            z1 = F.normalize(projection(features1), dim=1)
            z2 = F.normalize(projection(features2), dim=1)
            
            # Compute NT-Xent loss
            batch_size = z1.size(0)
            z = torch.cat([z1, z2], dim=0)
            sim_matrix = torch.matmul(z, z.T) / temperature
            
            # Mask out self-comparisons
            mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
            sim_matrix.masked_fill_(mask, -float("inf"))
            
            # Create positive mask
            labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)], dim=0).to(device)
            pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
            
            # Compute loss
            exp_sim = torch.exp(sim_matrix)
            pos_sim = exp_sim * pos_mask.float()
            loss = -torch.log(pos_sim.sum(dim=1) / exp_sim.sum(dim=1)).mean()
            
            optimizer.zero_grad()
            proj_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            proj_optimizer.step()
            
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(pretrain_loader)
        print(f"Contrastive Pretraining Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
                
    # Save the feature extractor
    print("Saving pretrained feature extractor...")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
    
    return model

