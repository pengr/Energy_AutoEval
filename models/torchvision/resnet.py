import torch.nn as nn
import torchvision.models as models


class ResNet(nn.Module):
    def __init__(self, base_model, pretrained, num_classes):
        super(ResNet, self).__init__()

        if base_model == "ResNet18":
            self.backbone = models.resnet18(pretrained=pretrained, num_classes=1000)

        if base_model == "ResNet34":
            self.backbone = models.resnet34(pretrained=pretrained, num_classes=1000)

        if base_model == "ResNet50":
            self.backbone = models.resnet50(pretrained=pretrained, num_classes=1000)

        if base_model == "ResNet101":
            self.backbone = models.resnet101(pretrained=pretrained, num_classes=1000)

        if base_model == "ResNet152":
            self.backbone = models.resnet152(pretrained=pretrained, num_classes=1000)

        if base_model == "Wide_ResNet50_2":
            self.backbone = models.wide_resnet50_2(pretrained=pretrained, num_classes=1000)

        if base_model == "Wide_ResNet101_2":
            self.backbone = models.wide_resnet101_2(pretrained=pretrained, num_classes=1000)

        # need to modify ResNet's fc
        assert self.backbone, "backbone cannot be None"
        if num_classes != 1000:
            dim_mlp = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(dim_mlp, num_classes)

    def forward(self, x):
        return self.backbone(x)

def ResNet18(base_model, pretrained, num_classes): return ResNet(base_model, pretrained, num_classes)
def ResNet34(base_model, pretrained, num_classes): return ResNet(base_model, pretrained, num_classes)
def ResNet50(base_model, pretrained, num_classes): return ResNet(base_model, pretrained, num_classes)
def ResNet101(base_model, pretrained, num_classes): return ResNet(base_model, pretrained, num_classes)
def ResNet152(base_model, pretrained, num_classes): return ResNet(base_model, pretrained, num_classes)
def Wide_ResNet50_2(base_model, pretrained, num_classes): return ResNet(base_model, pretrained, num_classes)
def Wide_ResNet101_2(base_model, pretrained, num_classes): return ResNet(base_model, pretrained, num_classes)
