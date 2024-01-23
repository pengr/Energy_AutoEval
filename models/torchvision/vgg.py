import torch.nn as nn
import torchvision.models as models


class VGG(nn.Module):
    def __init__(self, base_model, pretrained, num_classes):
        super(VGG, self).__init__()

        if base_model == "VGG11":
            self.backbone = models.vgg11(pretrained=pretrained, num_classes=1000)

        if base_model == "VGG11_bn":
            self.backbone = models.vgg11_bn(pretrained=pretrained, num_classes=1000)

        if base_model == "VGG13":
            self.backbone = models.vgg13(pretrained=pretrained, num_classes=1000)

        if base_model == "VGG13_bn":
            self.backbone = models.vgg13_bn(pretrained=pretrained, num_classes=1000)

        if base_model == "VGG16":
            self.backbone = models.vgg16(pretrained=pretrained, num_classes=1000)

        if base_model == "VGG16_bn":
            self.backbone = models.vgg16_bn(pretrained=pretrained, num_classes=1000)

        if base_model == "VGG19":
            self.backbone = models.vgg19(pretrained=pretrained, num_classes=1000)

        if base_model == "VGG19_bn":
            self.backbone = models.vgg19_bn(pretrained=pretrained, num_classes=1000)

        # need to modify VGG's fc
        assert self.backbone, "backbone cannot be None"
        if num_classes != 1000:
            dim_mlp = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(dim_mlp, num_classes)

    def forward(self, x):
        return self.backbone(x)

def VGG11(base_model, pretrained, num_classes): return VGG(base_model, pretrained, num_classes)
def VGG11_bn(base_model, pretrained, num_classes): return VGG(base_model, pretrained, num_classes)
def VGG13(base_model, pretrained, num_classes): return VGG(base_model, pretrained, num_classes)
def VGG13_bn(base_model, pretrained, num_classes): return VGG(base_model, pretrained, num_classes)
def VGG16(base_model, pretrained, num_classes): return VGG(base_model, pretrained, num_classes)
def VGG16_bn(base_model, pretrained, num_classes): return VGG(base_model, pretrained, num_classes)
def VGG19(base_model, pretrained, num_classes): return VGG(base_model, pretrained, num_classes)
def VGG19_bn(base_model, pretrained, num_classes): return VGG(base_model, pretrained, num_classes)
