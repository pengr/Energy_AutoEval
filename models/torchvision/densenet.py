import torch.nn as nn
import torchvision.models as models


class DenseNet(nn.Module):
    def __init__(self, base_model, pretrained, num_classes):
        super(DenseNet, self).__init__()

        if base_model == "DenseNet121":
            self.backbone = models.densenet121(pretrained=pretrained, num_classes=1000)

        if base_model == "DenseNet161":
            self.backbone = models.densenet161(pretrained=pretrained, num_classes=1000)

        if base_model == "DenseNet169":
            self.backbone = models.densenet169(pretrained=pretrained, num_classes=1000)

        if base_model == "DenseNet201":
            self.backbone = models.densenet201(pretrained=pretrained, num_classes=1000)

        # need to modify DenseNet's fc
        assert self.backbone, "backbone cannot be None"
        if num_classes != 1000:
            dim_mlp = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Linear(dim_mlp, num_classes)

    def forward(self, x):
        return self.backbone(x)

def DenseNet121(base_model, pretrained, num_classes): return DenseNet(base_model, pretrained, num_classes)
def DenseNet161(base_model, pretrained, num_classes): return DenseNet(base_model, pretrained, num_classes)
def DenseNet169(base_model, pretrained, num_classes): return DenseNet(base_model, pretrained, num_classes)
def DenseNet201(base_model, pretrained, num_classes): return DenseNet(base_model, pretrained, num_classes)
