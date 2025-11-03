import torch
import torchvision

# DORN paper: https://arxiv.org/abs/1806.02446
class DeepLab(torch.nn.Module):
    def __init__(self, K):
        super().__init__()
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
        self.model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(
            2048, K)

    def forward(self, x):
        return self.model(x)['out']

class SDNet(torch.nn.Module):
    def __init__(self, K, baseline):
        super().__init__()
        self.baseline = baseline
        # Initialize DeepLabV3 with pre-trained weights
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')

        # Freeze parameters of the first and second ResNet blocks
        for name, param in self.model.backbone.named_parameters():
            if 'layer1' in name or 'layer2' in name:
                param.requires_grad = False

        if self.baseline:
             # Single-channel regression head
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Conv2d(2048, K, kernel_size=1),  # Output a single channel
            )

        # Classifier with nclasses head and dropout
        else:
            self.model.classifier = torch.nn.Sequential(
                torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, K),
                torch.nn.Dropout(p=0.5)
                )

    def forward(self, x):
        out = self.model(x)['out']
        if self.baseline:
            out = 2 + torch.nn.functional.softplus(out)
            # out = 2 + torch.sigmoid(out) * (80 - 2)  # Scale to [2, 80]
        return out
