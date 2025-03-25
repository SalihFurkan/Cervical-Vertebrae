import torch
import torch.nn as nn
import torch.nn.functional as F

class RefinementClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Softmax(dim=1)  # Output 0 (keep Top-1) or 1 (switch to Top-2)
        )
        
    def forward(self, x):
        return self.fc(x) # Returns probability of switching to Top-2
