import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import matplotlib.pyplot as plt
from yolov2.datasets.voc import VOCDataset
import numpy as np
import torch
import matplotlib.patches as patches

def visualize_dataset(dataset, num_samples=5):
    """
    Hiển thị một số mẫu từ dataset để kiểm tra.
    Args:
        dataset (Dataset): Dataset cần kiểm tra.
        num_samples (int): Số mẫu hiển thị.
    """
    for i in range(num_samples):
        # Lấy dữ liệu từ dataset
        image, bboxes = dataset[i]
        
        # Chuyển Tensor sang NumPy array nếu cần
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()  # Từ CxHxW sang HxWxC
        
        # Kiểm tra kiểu dữ liệu và đảm bảo đúng kiểu
        if image.max() <= 1.0:  # Nếu ảnh trong khoảng [0, 1]
            image = (image * 255).astype(np.uint8)
        
        # Tạo figure để hiển thị ảnh
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(image)

        # Vẽ bounding boxes
        for bbox in bboxes:
            xmin, ymin, xmax, ymax, label = bbox
            xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                xmin, ymin - 10, str(label),
                color='red', fontsize=12, backgroundcolor='white'
            )

        # Hiển thị ảnh
        plt.axis('off')
        plt.show()


def main():
    # Cấu hình dataset VOC 
    voc_root = "./VOCdevkit/VOC2007"  # Thay đổi đường dẫn nếu cần
    voc_dataset = VOCDataset(
        root=voc_root,
        image_set="trainval",
        input_size=416)
    
    print(f"Loaded VOC dataset with {len(voc_dataset)} samples.")

    # Hiển thị mẫu từ dataset VOC
    print("Visualizing samples from VOC dataset:")
    visualize_dataset(voc_dataset, num_samples=5)

if __name__ == "__main__":
    main()
