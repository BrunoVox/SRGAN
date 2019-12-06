import torch
import torch.nn as nn
import models.modules.submodules as sm
from math import log10

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, nf, num_resblocks, upscale_factor, norm_type='batch', act_type='prelu', init_weights=True):
        super(Generator, self).__init__()
        firstConv = sm.Conv2dBlock(
            input_nc=input_nc,
            output_nc=nf,
            kernel_size=9,
            norm_type=None,
            act_type=act_type
        )
        resBlocks = [
            sm.ResNetBlock(
                input_nc=nf,
                mid_nc=nf,
                output_nc=nf,
                kernel_size=3,
                norm_type=norm_type,
                act_type=act_type
            ) for _ in range(num_resblocks)
        ]
        secondConv = sm.Conv2dBlock(
            input_nc=nf,
            output_nc=nf,
            kernel_size=3,
            norm_type=norm_type,
            act_type=None
        )
        upscaleBlock = [
            sm.SubPixelConvBlock(
                input_nc=nf,
                output_nc=nf,
                feature_factor=2,
                kernel_size=3,
                norm_type=None,
                act_type=act_type
            ) for _ in range(int(log10(upscale_factor) / log10(2)))
        ]
        finalConv = sm.Conv2dBlock(
            input_nc=nf,
            output_nc=output_nc,
            kernel_size=9,
            norm_type=None,
            act_type=None
        )

        self.beforeUp = nn.Sequential(firstConv, sm.ShortcutBlock(nn.Sequential(*resBlocks, secondConv)))
        self.up = nn.Sequential(*upscaleBlock)
        self.afterUp = finalConv

        if init_weights:
            self._initialize_weights()

        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.beforeUp(x)
        x = self.up(x)
        x = self.afterUp(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    # def check_init(self, mean=0.0, std=2e-2, valw=1.0, valb=0.0):
    #     check = True
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             if m.weight.mean().round().abs().item() != mean or (m.weight.std() * 100).round().item() != std * 100:
    #                 check = False
    #                 break
    #             if m.bias is not None:
    #                 if m.bias.eq(valb).sum().item() != m.bias.size(0):
    #                     check = False
    #                     break

    #         elif isinstance(m, nn.BatchNorm2d):
    #             if m.weight.eq(valw).sum().item() != m.weight.size(0):
    #                 check = False
    #                 break
    #             if m.bias is not None:
    #                 if m.bias.eq(valb).sum().item() != m.bias.size(0):
    #                     check = False
    #                     break
    #     if check:
    #         print('Weights and biases initialized correctly!')
    #     else:
    #         print('There was a problem initializing weight(s) or bias(es)...')

    