import torch
import torch.nn as nn

class YOLOv2Loss(nn.Module):
    """
    YOLOv2 Loss:
    - Modular implementation with separate loss functions.
    """
    def __init__(
        self,
        S=13,
        B=5,
        num_classes=20,
        anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053), (11.2364, 10.0071)],
        lambda_coord=5,
        lambda_noobj=0.5
    ):
        super(YOLOv2Loss, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.anchors = anchors
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def build_grid(self, pred):
        """
        Build grid offsets (cx, cy) for each cell.
        Shape: (S, S, 1, 2) -> Broadcast to (N, S, S, B, 2)
        """
        S = pred.size(1)
        grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S))
        grid_x = grid_x.unsqueeze(-1).float()
        grid_y = grid_y.unsqueeze(-1).float()
        grid = torch.cat((grid_x, grid_y), dim=-1)
        return grid.to(pred.device)

    def transform_preds_to_bbox(self, pred):
        """
        Transform (tx, ty, tw, th) into (x1, y1, x2, y2) for IoU calculation.
        """
        tx, ty, tw, th = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
        anchors = torch.tensor(self.anchors, dtype=torch.float32, device=pred.device)
        anchors = anchors.view(1, 1, 1, self.B, 2)
        grid = self.build_grid(pred).unsqueeze(0).unsqueeze(3)
        bx = (torch.sigmoid(tx) + grid[..., 0]) / self.S
        by = (torch.sigmoid(ty) + grid[..., 1]) / self.S
        bw = (anchors[..., 0] * torch.exp(tw)) / self.S
        bh = (anchors[..., 1] * torch.exp(th)) / self.S
        x1, y1, x2, y2 = bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2
        return x1, y1, x2, y2

    def compute_iou(self, pred_boxes, target_boxes):
        """
        Compute IoU between predicted and target boxes.
        """
        inter_x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
        inter_y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
        inter_x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
        inter_y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        tgt_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
        union_area = pred_area + tgt_area - inter_area + 1e-16
        return inter_area / union_area

    def compute_coord_loss(self, obj_mask, tx_p, ty_p, tw_p, th_p, tx_t, ty_t, tw_t, th_t):
        """
        Compute coordinate loss for bounding box predictions.
        """
        coord_loss = self.lambda_coord * torch.sum(
            obj_mask * ((tx_p - tx_t) ** 2 + (ty_p - ty_t) ** 2)
        )
        coord_loss += self.lambda_coord * torch.sum(
            obj_mask * ((tw_p - tw_t) ** 2 + (th_p - th_t) ** 2)
        )
        return coord_loss

    def compute_conf_loss(self, obj_mask, noobj_mask, conf_p, conf_target):
        """
        Compute confidence loss for object and no-object predictions.
        """
        conf_loss_obj = torch.sum(obj_mask * ((conf_p - conf_target) ** 2))
        conf_loss_noobj = self.lambda_noobj * torch.sum(noobj_mask * (conf_p ** 2))
        return conf_loss_obj + conf_loss_noobj

    def compute_class_loss(self, obj_mask, class_p, class_t):
        """
        Compute class prediction loss.
        """
        return torch.sum(obj_mask.unsqueeze(-1) * ((class_p - class_t) ** 2))

    def forward(self, predictions, targets):
        """
        Compute total YOLOv2 loss.
        """
        tx_p, ty_p, tw_p, th_p, conf_p, class_p = predictions[..., 0], predictions[..., 1], predictions[..., 2], predictions[..., 3], predictions[..., 4], predictions[..., 5:]
        tx_t, ty_t, tw_t, th_t, conf_t, class_t = targets[..., 0], targets[..., 1], targets[..., 2], targets[..., 3], targets[..., 4], targets[..., 5:]

        # Transform predictions and targets to (x1, y1, x2, y2)
        x1_p, y1_p, x2_p, y2_p = self.transform_preds_to_bbox(predictions)
        pred_boxes = torch.stack([x1_p, y1_p, x2_p, y2_p], dim=-1)

        pred_like_target = torch.cat([tx_t.unsqueeze(-1), ty_t.unsqueeze(-1), tw_t.unsqueeze(-1), th_t.unsqueeze(-1), conf_t.unsqueeze(-1), class_t], dim=-1)
        x1_t, y1_t, x2_t, y2_t = self.transform_preds_to_bbox(pred_like_target)
        target_boxes = torch.stack([x1_t, y1_t, x2_t, y2_t], dim=-1)

        # Object and no-object masks
        obj_mask = (conf_t > 0).float()
        noobj_mask = 1.0 - obj_mask

        # IoU for confidence target
        iou = self.compute_iou(pred_boxes, target_boxes)
        conf_target = iou * obj_mask

        # Compute individual losses
        coord_loss = self.compute_coord_loss(obj_mask, tx_p, ty_p, tw_p, th_p, tx_t, ty_t, tw_t, th_t)
        conf_loss = self.compute_conf_loss(obj_mask, noobj_mask, conf_p, conf_target)
        class_loss = self.compute_class_loss(obj_mask, class_p, class_t)

        # Total loss
        total_loss = coord_loss + conf_loss + class_loss
        return total_loss
