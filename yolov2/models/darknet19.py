import torch
import torch.nn as nn

class Darknet19(nn.Module):
    def __init__(self):
        super(Darknet19, self).__init__()
        self.features_passthrough = nn.Sequential(
            # Block 1
            self._conv_block(3, 32, 3),
            nn.MaxPool2d(2, 2),

            # Block 2
            self._conv_block(32, 64, 3),
            nn.MaxPool2d(2, 2),

            # Block 3
            self._conv_block(64, 128, 3),
            self._conv_block(128, 64, 1),
            self._conv_block(64, 128, 3),
            nn.MaxPool2d(2, 2),

            # Block 4
            self._conv_block(128, 256, 3),
            self._conv_block(256, 128, 1),
            self._conv_block(128, 256, 3),
            nn.MaxPool2d(2, 2),

            # Block 5 (features for passthrough layer)
            self._conv_block(256, 512, 3),
            self._conv_block(512, 256, 1),
            self._conv_block(256, 512, 3),
            self._conv_block(512, 256, 1),
            self._conv_block(256, 512, 3)
        )

        self.features_final = nn.Sequential(
            nn.MaxPool2d(2, 2),  # Downsample for Block 6
            self._conv_block(512, 1024, 3),
            self._conv_block(1024, 512, 1),
            self._conv_block(512, 1024, 3),
            self._conv_block(1024, 512, 1),
            self._conv_block(512, 1024, 3)
        )

    def _conv_block(self, in_channels, out_channels, kernel_size):
        """Convolutional block with BatchNorm and LeakyReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        passthrough_features = self.features_passthrough(x)  # 26x26x512
        final_features = self.features_final(passthrough_features)  # 13x13x1024
        return passthrough_features, final_features
