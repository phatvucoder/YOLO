import torch

def intersection_over_union(boxes_preds, boxes_labels):
    """
        box format: [x1,y1,x2,y2]
    """
    box1_x1 = boxes_preds[..., 0]
    box1_y1 = boxes_preds[..., 1]
    box1_x2 = boxes_preds[..., 2]
    box1_y2 = boxes_preds[..., 3]

    box2_x1 = boxes_labels[..., 0]
    box2_y1 = boxes_labels[..., 1]
    box2_x2 = boxes_labels[..., 2]
    box2_y2 = boxes_labels[..., 3]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = (box1_x2 - box1_x1).clamp(0) * (box1_y2 - box1_y1).clamp(0)
    box2_area = (box2_x2 - box2_x1).clamp(0) * (box2_y2 - box2_y1).clamp(0)

    union = box1_area + box2_area - intersection + 1e-6
    iou = intersection / union
    return iou

def non_max_suppression(predictions, iou_threshold=0.5, threshold=0.4):
    """
    predictions: list of [x1,y1,x2,y2,score]
    """
    if len(predictions) == 0:
        return []

    predictions = [pred for pred in predictions if pred[4] > threshold]
    predictions.sort(key=lambda x: x[4], reverse=True)

    filtered_boxes = []
    while predictions:
        chosen_box = predictions.pop(0)
        filtered_boxes.append(chosen_box)

        preds_after = []
        for box in predictions:
            iou_val = intersection_over_union(
                torch.tensor(chosen_box[:4]).unsqueeze(0),
                torch.tensor(box[:4]).unsqueeze(0)
            )
            if iou_val < iou_threshold:
                preds_after.append(box)
        predictions = preds_after

    return filtered_boxes
