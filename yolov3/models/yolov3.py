import torch.nn as nn
import torch
from models.darknet53 import Darknet53
from models.yololayer import YOLOLayer

class YOLOv3(nn.Module):
    def __init__(self, num_classes=80, anchors=None):
        super(YOLOv3, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.backbone = Darknet53()

        # Prediction layers cho feature map 13x13
        self.prediction1 = self._make_prediction_layer(1024, 512)
        self.yolo1 = YOLOLayer(512, 3, num_classes)

        # Upsample và các lớp bổ sung cho feature map 26x26
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv26 = self._make_conv_layer(512, 256, 1)
        self.prediction2 = self._make_prediction_layer(768, 256)  # 512 (upsample) + 256 (conv) = 768
        self.yolo2 = YOLOLayer(256, 3, num_classes)

        # Upsample và các lớp bổ sung cho feature map 52x52
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv52 = self._make_conv_layer(256, 128, 1)
        self.prediction3 = self._make_prediction_layer(384, 128)  # 256 (upsample) + 128 (conv) = 384
        self.yolo3 = YOLOLayer(128, 3, num_classes)

    def _make_conv_layer(self, in_channels, out_channels, kernel_size):
        """Lớp convolution đơn giản dùng để giảm số kênh trước khi concatenate"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def _make_prediction_layer(self, in_channels, out_channels):
        """Tạo prediction block trước YOLO layer"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                out_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
        )

    def forward(self, x):
        # Trích xuất các feature maps từ backbone
        feat13, feat26, feat52 = self.backbone(x)

        # Dự đoán cho feature map 13x13
        yolo1_input = self.prediction1(feat13)
        yolo1_output = self.yolo1(yolo1_input)

        # Dự đoán cho feature map 26x26
        upsample1 = self.upsample1(yolo1_input)
        conv26 = self.conv26(feat26)
        concat26 = torch.cat((upsample1, conv26), dim=1)  # 512 (upsample) + 256 (conv) = 768
        yolo2_input = self.prediction2(concat26)
        yolo2_output = self.yolo2(yolo2_input)

        # Dự đoán cho feature map 52x52
        upsample2 = self.upsample2(yolo2_input)
        conv52 = self.conv52(feat52)
        concat52 = torch.cat((upsample2, conv52), dim=1)  # 256 (upsample) + 128 (conv) = 384
        yolo3_input = self.prediction3(concat52)
        yolo3_output = self.yolo3(yolo3_input)

        return yolo1_output, yolo2_output, yolo3_output

# # Test model
# x = torch.randn(1, 3, 416, 416)
# model = YOLOv3(num_classes=80)
# outputs = model(x)
# for i, output in enumerate(outputs, 1):
#     print(f"YOLO Layer {i} output shape: {output.shape}")