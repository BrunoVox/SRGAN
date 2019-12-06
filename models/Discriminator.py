import torch
import torch.nn as nn
import models.modules.submodules as sm

class Discriminator(nn.Module):
    def __init__(self, input_nc, nf, norm_type='batch', act_type='leaky_relu', init_weights=True):
        super(Discriminator, self).__init__()

        firstBlock = nn.Sequential(
            sm.Conv2dBlock(
                input_nc=input_nc,
                output_nc=nf,
                kernel_size=3,
                norm_type=None,
                act_type=act_type
            ),
            sm.Conv2dBlock(
                input_nc=input_nc,
                output_nc=nf,
                kernel_size=3,
                stride=2,
                norm_type=norm_type,
                act_type=act_type
            )
        )

        generalBlock = [
            nn.Sequential(
                sm.Conv2dBlock(
                    input_nc=nf * 2 ** (i - 1),
                    output_nc=nf * 2 ** i,
                    kernel_size=3,
                    norm_type=norm_type,
                    act_type=act_type
                ),
                sm.Conv2dBlock(
                    input_nc=nf * 2 ** i,
                    output_nc=nf * 2 ** i,
                    kernel_size=3,
                    stride=2,
                    norm_type=norm_type,
                    act_type=act_type
                )
            ) for i in range(1, 4)
        ]

        finalBlock = nn.Sequential(
            nn.Linear(in_features=512 * 6 * 6, out_features=1024),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=1024, out_features=1),
            nn.Sigmoid()
        )

        self.features = nn.Sequential(firstBlock, *generalBlock)
        self.classifier = finalBlock

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x).flatten(1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 2e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 2e-2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def check_init(self, mean=0.0, std=2e-2, valw=1.0, valb=0.0):
        check = True
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.weight.mean().round().abs().item() != mean or (m.weight.std() * 100).round().item() != std:
                    check = False
                    break
                if m.bias is not None:
                    if m.bias.eq(valb).sum().item() != m.bias.size(0):
                        check = False
                        break

            elif isinstance(m, nn.BatchNorm2d):
                if m.weight.eq(valw).sum().item() != m.weight.size(0):
                    check = False
                    break
                if m.bias is not None:
                    if m.bias.eq(valb).sum().item() != m.bias.size(0):
                        check = False
                        break

            elif isinstance(m, nn.Linear):
                if m.weight.mean().round().abs().item() != mean or (m.weight.std() * 100).round().item() != std:
                    check = False
                    break
                if m.bias is not None:
                    if m.bias.eq(valb).sum().item() != m.bias.size(0):
                        check = False
                        break
        if check:
            print('Weights and biases initialized correctly!')
        else:
            print('There was a problem initializing weight(s) or bias(es)...')

