import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, num_classes=6, alpha=1.0, beta=1.0, gamma=1.0, class_weights=None, soft_targets=None):
        """
        Initializes the combined loss function.
        
        Args:
            num_classes (int): Number of classes.
            alpha (float): Weight for Ordinal Loss.
            beta (float): Weight for Soft Targets Loss.
            gamma (float): Weight for Weighted CrossEntropy Loss.
            class_weights (tensor): Class weights for CrossEntropy.
            soft_targets (tensor): Soft target matrix for soft labels.
        """
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.class_weights = class_weights
        self.soft_targets = soft_targets
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()

    def ordinal_loss(self, outputs, targets):
        """
        Computes ordinal loss.
        """
        batch_size = targets.size(0)
        targets_one_hot = torch.zeros(batch_size, self.num_classes).to(targets.device)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        distance_matrix = torch.abs(torch.arange(self.num_classes, dtype=torch.float32).unsqueeze(0) -
                                    torch.arange(self.num_classes, dtype=torch.float32).unsqueeze(1)).to(targets.device)

        softmax_outputs = F.softmax(outputs, dim=1)
        penalty = torch.matmul(softmax_outputs, distance_matrix)
        loss = torch.sum(penalty * targets_one_hot, dim=1)
        
        loss = loss * self.class_weights[targets]  # Scale loss per target class

        return loss.mean()

    def soft_targets_loss(self, outputs, targets):
        """
        Computes soft target loss using soft labels.
        """
        log_probs = F.log_softmax(outputs, dim=1)
        soft_target_labels = self.soft_targets[targets]  # Retrieve soft targets for the given labels
        loss = - (soft_target_labels * log_probs).sum(dim=1).mean()
        return loss
        
    def ordinal_loss_with_soft_targets(self, outputs, targets):
        """
        Implements an ordinal regression loss that penalizes predictions
        that are further from the true class more heavily.
        """
        # Convert targets to one-hot encoding
        batch_size = targets.size(0)
        
        # Calculate distance penalty matrix
        distance_matrix = torch.abs(torch.arange(self.num_classes).unsqueeze(0) -
                                   torch.arange(self.num_classes).unsqueeze(1)).to(targets.device)
        distance_matrix = distance_matrix.to(dtype=torch.float32)
        
        # Calculate weighted loss
        softmax_outputs = F.softmax(outputs, dim=1)
        penalty = torch.matmul(softmax_outputs, distance_matrix)
        loss = torch.sum(penalty * self.soft_targets[targets], dim=1).mean()
        
        return loss

    def forward(self, outputs, targets):
        """
        Computes the total loss by combining all three loss functions.
        """
        ord_loss = self.ordinal_loss(outputs, targets) if self.alpha > 0 else 0
        soft_loss = self.soft_targets_loss(outputs, targets) if self.beta > 0 else 0
        ce_loss = self.cross_entropy(outputs, targets) if self.gamma > 0 else 0

        total_loss = (self.alpha * ord_loss) + (self.beta * soft_loss) + (self.gamma * ce_loss)
        return total_loss
    

