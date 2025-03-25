import torch
import torch.nn as nn

class OrdinalClassifier(nn.Module):
    def __init__(self, num_classes, device):
        super(OrdinalClassifier, self).__init__()
        self.num_classes = num_classes
        self.thresholds = nn.Parameter(torch.linspace(-2, 2, num_classes - 1).to(device))

    def forward(self, logits):
        cumulative_probs = torch.sigmoid(logits[:, :-1] - self.thresholds)
        probs = torch.zeros(logits.shape[0], self.num_classes).to(logits.device)
        probs[:, 0] = cumulative_probs[:, 0]
        for k in range(1, self.num_classes - 1):
            probs[:, k] = cumulative_probs[:, k] - cumulative_probs[:, k-1]
        probs[:, self.num_classes - 1] = 1.0 - cumulative_probs[:, self.num_classes - 2]
        probs = torch.clamp(probs, min=0.0)
        return probs / (probs.sum(dim=1, keepdim=True) + 1e-10)
