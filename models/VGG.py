import torch
import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, layers=19, bn=False, loss_config='VGG54', pretrained=True):
        super(VGG, self).__init__()
        if loss_config == 'VGG54':
            if not bn:
                if layers == 19:
                    model = models.vgg19(pretrained=pretrained).features[:36]
                elif layers == 16:
                    model = models.vgg16(pretrained=pretrained).features[:30]
                elif layers == 13:
                    model = models.vgg13(pretrained=pretrained).features[:24]
                elif layers == 11:
                    model = models.vgg11(pretrained=pretrained).features[:20]
            else:
                if layers == 19:
                    model = models.vgg19_bn(pretrained=pretrained).features[:52]
                elif layers == 16:
                    model = models.vgg16_bn(pretrained=pretrained).features[:43]
                elif layers == 13:
                    model = models.vgg13_bn(pretrained=pretrained).features[:34]
                elif layers == 11:
                    model = models.vgg11_bn(pretrained=pretrained).features[:28]

        elif loss_config == 'VGG22':
            if not bn:
                if layers == 19:
                    model = models.vgg19(pretrained=pretrained).features[:9]
                elif layers == 16:
                    model = models.vgg16(pretrained=pretrained).features[:9]
                elif layers == 13:
                    model = models.vgg13(pretrained=pretrained).features[:9]
                elif layers == 11:
                    model = models.vgg11(pretrained=pretrained).features[:5]
            else:
                if layers == 19:
                    model = models.vgg19_bn(pretrained=pretrained).features[:13]
                elif layers == 16:
                    model = models.vgg16_bn(pretrained=pretrained).features[:13]
                elif layers == 13:
                    model = models.vgg13_bn(pretrained=pretrained).features[:13]
                elif layers == 11:
                    model = models.vgg11_bn(pretrained=pretrained).features[:7]
        if pretrained:
            for param in model.parameters():
                param.requires_grad = False
        self.model = model

        mean = torch.Tensor([0.485 * 2 - 1, 0.456 * 2 - 1, 0.406 * 2 - 1]).view(1,3,1,1)
        std = torch.Tensor([0.229 * 2, 0.224 * 2, 0.225 * 2]).view(1,3,1,1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.model(x) / 12.75
        return x

        