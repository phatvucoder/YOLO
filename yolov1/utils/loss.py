import torch
import torch.nn as nn
from .nms import intersection_over_union

class YOLOLoss(nn.Module):
    def __init__(self, gridSize=7, numBoxes=2, numClasses=20, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.gridSize = gridSize
        self.numBoxes = numBoxes
        self.numClasses = numClasses
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        batchSize = predictions.size(0)

        # (N, S, S, 5B + C)
        predictions = predictions.view(batchSize, self.gridSize, self.gridSize, self.numBoxes*5 + self.numClasses)
        targets = targets.view(batchSize, self.gridSize, self.gridSize, self.numBoxes*5 + self.numClasses)

        # Box
        pred_box = predictions[..., :self.numBoxes*5].view(batchSize, self.gridSize, self.gridSize, self.numBoxes, 5)
        pred_xy = pred_box[..., 0:2]
        pred_wh = pred_box[..., 2:4]
        pred_conf = pred_box[..., 4]

        tgt_box = targets[..., :self.numBoxes*5].view(batchSize, self.gridSize, self.gridSize, self.numBoxes, 5)
        tgt_xy = tgt_box[..., 0:2]
        tgt_wh = tgt_box[..., 2:4]
        tgt_conf = tgt_box[..., 4]

        # Class
        pred_class = predictions[..., self.numBoxes*5:]
        tgt_class = targets[..., self.numBoxes*5:]

        # Calc IoU & select bbox responsible
        ious = []
        for b_i in range(self.numBoxes):
            iou_b = intersection_over_union(
                xywh_to_xyxy(pred_xy[..., b_i, :], pred_wh[..., b_i, :]),
                xywh_to_xyxy(tgt_xy[..., b_i, :], tgt_wh[..., b_i, :])
            )
            ious.append(iou_b.unsqueeze(-1))  # (N,S,S,1)
        ious = torch.cat(ious, dim=-1)  # (N,S,S,B)
        iou_max, best_box_idx = ious.max(dim=-1)  # (N,S,S), (N,S,S)

        # responsible_mask
        # => one_hot: (N,S,S,B)
        responsible_mask = nn.functional.one_hot(best_box_idx, num_classes=self.numBoxes).float()

        # Sqrt w,h
        pred_wh_sqrt = torch.sign(pred_wh) * torch.sqrt(torch.abs(pred_wh) + 1e-6)
        tgt_wh_sqrt = torch.sqrt(tgt_wh + 1e-6)

        # (x, y) loss => only bboxes responsible
        # => shape broadcast
        resp_mask_xywh = responsible_mask.unsqueeze(-1) # (N,S,S,B,1)
        coord_loss_xy = self.mse(resp_mask_xywh * pred_xy, resp_mask_xywh * tgt_xy)
        coord_loss_wh = self.mse(resp_mask_xywh * pred_wh_sqrt, resp_mask_xywh * tgt_wh_sqrt)
        coord_loss = (coord_loss_xy + coord_loss_wh) * self.lambda_coord

        # Object confidence loss => only bboxes responsible
        resp_mask_conf = responsible_mask
        obj_loss = self.mse(resp_mask_conf * pred_conf, resp_mask_conf * tgt_conf)

        # No-object => bboxes not responsible
        noobj_mask = 1 - responsible_mask
        noobj_loss = self.mse(noobj_mask * pred_conf, noobj_mask * tgt_conf) * self.lambda_noobj

        # Class loss => cell has obj (tgt_conf > 0). 
        cell_obj_mask = (tgt_conf[..., 0] > 0).float()  # (N,S,S)
        cell_obj_mask = cell_obj_mask.unsqueeze(-1).expand_as(pred_class)
        class_loss = self.mse(cell_obj_mask * pred_class, cell_obj_mask * tgt_class)

        total_loss = coord_loss + obj_loss + noobj_loss + class_loss
        total_loss = total_loss / batchSize
        return total_loss

def xywh_to_xyxy(xy, wh):
    # xy, wh shape = (N,S,S,2)
    x1 = xy[...,0] - wh[...,0]/2
    y1 = xy[...,1] - wh[...,1]/2
    x2 = xy[...,0] + wh[...,0]/2
    y2 = xy[...,1] + wh[...,1]/2
    return torch.stack([x1, y1, x2, y2], dim=-1)
