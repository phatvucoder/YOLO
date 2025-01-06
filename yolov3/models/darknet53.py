import torch.nn as nn
from models.residualblock import ResidualBlock

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        self.layer1 = self._conv_block(3, 32, 3)
        self.layer2 = self._conv_block(32, 64, 3, stride=2)
        self.residual1 = self._residual_block(64, 64, num_blocks=1)

        self.layer3 = self._conv_block(64, 128, 3, stride=2)
        self.residual2 = self._residual_block(128, 128, num_blocks=2)

        self.layer4 = self._conv_block(128, 256, 3, stride=2)
        self.residual3 = self._residual_block(256, 256, num_blocks=8)

        self.layer5 = self._conv_block(256, 512, 3, stride=2)
        self.residual4 = self._residual_block(512, 512, num_blocks=8)

        self.layer6 = self._conv_block(512, 1024, 3, stride=2)
        self.residual5 = self._residual_block(1024, 1024, num_blocks=4)

    def _conv_block(self, in_channels, out_channels, kernel_size, stride=1):
        """Convolutional block với BatchNorm và LeakyReLU"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def _residual_block(self, in_channels, out_channels, num_blocks):
        """Stack của Residual Blocks"""
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.residual1(x)

        x = self.layer3(x)
        x = self.residual2(x)

        x = self.layer4(x)
        x = self.residual3(x)
        feat52 = x  # Feature map 52x52

        x = self.layer5(x)
        x = self.residual4(x)
        feat26 = x  # Feature map 26x26

        x = self.layer6(x)
        x = self.residual5(x)
        feat13 = x  # Feature map 13x13

        return feat13, feat26, feat52


# # Test
# x = torch.randn(1, 3, 416, 416)
# model = Darknet53()
# outputs = model(x)
# for i, output in enumerate(outputs, 1):
#     print(f"Output {i} shape:", output.shape)
