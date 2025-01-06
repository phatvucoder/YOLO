import torch.nn as nn
import torch.nn.functional as F

class YOLOv3Loss(nn.Module):
    def __init__(self, num_classes=20, anchors=None, stride=32, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOv3Loss, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.stride = stride
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def forward(self, predictions, targets, input_size=416):
        obj_mask = targets[..., 4] == 1  # Object mask
        noobj_mask = targets[..., 4] == 0  # No-object mask

        # Confidence loss
        pred_conf = predictions[..., 4]
        true_conf = targets[..., 4]
        conf_loss_obj = F.binary_cross_entropy_with_logits(pred_conf[obj_mask], true_conf[obj_mask], reduction='sum')
        conf_loss_noobj = F.binary_cross_entropy_with_logits(pred_conf[noobj_mask], true_conf[noobj_mask], reduction='sum')
        conf_loss = conf_loss_obj + self.lambda_noobj * conf_loss_noobj

        # Class loss
        pred_cls = predictions[..., 5:]
        true_cls = targets[..., 5:]
        cls_loss = F.binary_cross_entropy_with_logits(pred_cls[obj_mask], true_cls[obj_mask], reduction='sum')

        # Coordinate loss
        pred_box = predictions[..., 0:4]
        true_box = targets[..., 0:4]
        coord_loss = F.mse_loss(pred_box[obj_mask], true_box[obj_mask], reduction='sum')

        # Total loss
        total_loss = self.lambda_coord * coord_loss + conf_loss + cls_loss
        return total_loss

# # Test loss
# import torch

# batch_size = 1
# grid_size = 13
# num_classes = 20
# anchors = [[116, 90], [156, 198], [373, 326]]

# # Create dummy predictions and targets
# predictions = torch.randn(batch_size, len(anchors), grid_size, grid_size, 5 + num_classes)
# targets = torch.zeros(batch_size, len(anchors), grid_size, grid_size, 5 + num_classes)

# # Add dummy ground-truth
# targets[0, 0, 5, 5, 0:4] = torch.tensor([0.5, 0.5, 1.0, 1.0])  # Bounding box
# targets[0, 0, 5, 5, 4] = 1.0  # Objectness
# targets[0, 0, 5, 5, 5] = 1.0  # Class ID 0 (one-hot)

# # Initialize loss function
# loss_fn = YOLOv3Loss(num_classes=num_classes, anchors=anchors, stride=32)

# # Compute loss
# loss = loss_fn(predictions, targets)
# print(f"Loss: {loss.item()}")
