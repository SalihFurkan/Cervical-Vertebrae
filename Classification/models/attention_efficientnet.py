import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .cbam import CBAM
from .ordinal_classifier import OrdinalClassifier
from .refinement_classifier import RefinementClassifier

class AttentionEfficientNet(nn.Module):
    def __init__(self, device, num_classes=6):
        super(AttentionEfficientNet, self).__init__()
        self.base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features = self.base_model.features
        self.cbam = CBAM(channels=1536)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1536, 512, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x, mode='classify', mask_roi=None):
        x = self.features(x)
        if mode == 'reconstruct':
            return self.forward_reconstruct(x)
        else:
            if mask_roi is not None:
                return self.forward_classify_with_roi(x, mask_roi)
            else:
                return self.forward_classify(x)
        
    def forward_reconstruct(self, x):
        x = self.cbam(x)
        x = self.decoder(x)
        return x
    
    def forward_classify(self, x):
        x = self.cbam(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
        
    def forward_classify_with_roi(self, x, mask_roi):
        mask_roi = F.interpolate(mask_roi, size=(x.size()[2], x.size()[3]), mode='bilinear')[:,0,:,:].squeeze(1)
        spatial_attention = self.cbam.spatial_attention(x)
        spatial_attention = (spatial_attention - spatial_attention.min()) / (spatial_attention.max() - spatial_attention.min() + 1e-10)
        spatial_attention = spatial_attention.squeeze(1)
        roi_loss = F.mse_loss(spatial_attention, mask_roi)
        
        x = self.cbam(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x, roi_loss

