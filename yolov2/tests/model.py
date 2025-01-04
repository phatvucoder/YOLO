import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from yolov2.models.yolov2 import YOLOv2

# Khởi tạo mô hình YOLOv2 với các anchors giả định
num_classes = 10
anchors = [[ 97.64029008, 153.26487029],
 [145.69979857, 301.49219496],
 [341.08786389, 352.00467151],
 [291.4388784,  186.92711247],
 [ 39.52848088,  58.76526519]]  # Ví dụ danh sách anchors
model = YOLOv2(num_classes=num_classes, anchors=anchors)

# Tạo một tensor đầu vào giả với kích thước (batch_size, channels, height, width)
input_tensor = torch.randn(10, 3, 416, 416)  # Batch size = 1, ảnh RGB 416x416

# Truyền tensor qua mô hình
output = model(input_tensor)

# Kiểm tra kích thước đầu vào và đầu ra
print("Kích thước đầu vào:", input_tensor.size())
print("Kích thước đầu ra:", output.size())