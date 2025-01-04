import torch
import torch.nn as nn
from models.darknet19 import Darknet19

class YOLOv2(nn.Module):
    def __init__(self, num_classes=20, anchors=[]):
        super(YOLOv2, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = len(anchors)
        self.backbone = Darknet19()

        # Passthrough Layer
        self.passthrough_layer = nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0)

        # YOLO Head: Two separate heads
        self.bbox_conf_head = nn.Conv2d(1024 + 2048, self.num_anchors * 5, kernel_size=1)  # 5 values per anchor box
        self.class_prob_head = nn.Conv2d(1024 + 2048, self.num_anchors * self.num_classes, kernel_size=1)  # Class probabilities

    def forward(self, x):
        # Extract features from backbone
        passthrough_features, final_features = self.backbone(x)

        # Downsample passthrough features to 13x13
        passthrough_features = nn.functional.max_pool2d(passthrough_features, kernel_size=2, stride=2)
        passthrough_features = self.passthrough_layer(passthrough_features)

        # Combine passthrough features and final features
        combined = torch.cat([final_features, passthrough_features], dim=1)  # 13x13x(1024+2048)

        # YOLO Head
        bbox_conf = self.bbox_conf_head(combined)  # (Batch, num_anchors * 5, 13, 13)
        class_probs = self.class_prob_head(combined)  # (Batch, num_anchors * num_classes, 13, 13)

        # Reshape outputs
        batch_size = x.size(0)
        bbox_conf = bbox_conf.view(batch_size, self.num_anchors, 5, 13, 13)  # (Batch, num_anchors, 5, 13, 13)
        class_probs = class_probs.view(batch_size, self.num_anchors, self.num_classes, 13, 13)  # (Batch, num_anchors, num_classes, 13, 13)

        # Concatenate for final output
        output = torch.cat([bbox_conf, class_probs], dim=2)  # (Batch, num_anchors, 5 + num_classes, 13, 13)

        # Permute to make the output shape: (Batch, 13, 13, num_anchors, 5 + num_classes)
        output = output.permute(0, 3, 4, 1, 2).contiguous()  # (Batch, 13, 13, num_anchors, 5 + num_classes)

        return output