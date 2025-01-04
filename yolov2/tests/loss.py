import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from yolov2.utils.loss import YOLOv2Loss

if __name__ == "__main__":
    # Giả sử output của model: (N, S, S, B, 5+num_classes)
    # S=13, B=5, num_classes=20
    N = 2
    S = 13
    B = 5
    C = 20
    predictions = torch.randn(N, S, S, B, 5 + C)
    targets = torch.randn(N, S, S, B, 5 + C)

    criterion = YOLOv2Loss(
        S=13,
        B=5,
        num_classes=20,
        anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892),
                 (9.47112, 4.84053), (11.2364, 10.0071)],
        lambda_coord=5,
        lambda_noobj=0.5
    )

    loss = criterion(predictions, targets)
    print("Loss:", loss.item())
