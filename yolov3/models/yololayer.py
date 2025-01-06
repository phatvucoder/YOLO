import torch.nn as nn

class YOLOLayer(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes):
        super(YOLOLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, num_anchors * (5 + num_classes), kernel_size=1, stride=1, padding=0
        )
        self.num_classes = num_classes

    def forward(self, x):
        batch_size, _, grid_size, _ = x.size()
        num_outputs = self.conv.out_channels
        num_anchors = num_outputs // (5 + self.num_classes)
        x = self.conv(x)
        return x.view(batch_size, num_anchors, 5 + self.num_classes, grid_size, grid_size).permute(0, 3, 4, 1, 2) # [B, H, W, num_anchors, 5 + num_classes]
