import os
import json
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class COCODataset(Dataset):
    def __init__(self, root, annotation_file, input_size=416, transform=None):
        self.root = root  # Root path to COCO images
        self.input_size = input_size  # Input image size
        self.transform = transform  # Image transform

        # Read annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)
        
        self.image_info = self.annotations['images']
        self.annotation_info = self.annotations['annotations']
        self.categories = {cat['id']: cat['name'] for cat in self.annotations['categories']}

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, index):
        """
        Get image and annotations.
        Args:
            index (int): Index of image.
        Returns:
            image (Tensor): Image. [C, H, W]
            bboxes (Tensor): Bounding boxes and labels. [N, 5]
        """
        image_info = self.image_info[index]
        image_path = os.path.join(self.root, image_info['file_name'])
        image_id = image_info['id']

        # Read image in BGR format
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read annotations (bounding boxes)
        bboxes = self.parse_coco_annotation(image_id)

        # Resize image and bounding boxes
        h, w, _ = image.shape
        image, bboxes = self.resize(image, bboxes, (self.input_size, self.input_size), (w, h))

        # Transform image
        if self.transform:
            image = self.transform(image)

        return image, bboxes

    def parse_coco_annotation(self, image_id):
        """
        Parse COCO annotation.
        Args:
            image_id (int): Image ID.
        Returns:
            bboxes (array): Bounding boxes and labels. [xmin, ymin, xmax, ymax, label]
        """
        bboxes = []
        for ann in self.annotation_info:
            if ann['image_id'] == image_id:
                xmin, ymin, width, height = ann['bbox']
                xmax = xmin + width
                ymax = ymin + height
                category_id = ann['category_id']
                label = self.categories[category_id]
                bboxes.append([xmin, ymin, xmax, ymax, category_id])
        return np.array(bboxes, dtype=torch.FloatTensor)

    def resize(self, image, bboxes, size, original_size):
        """
        Resize image and bounding boxes.
        Args:
            image (array): Image.
            bboxes (array): Bounding boxes and labels. [xmin, ymin, xmax, ymax, label]
            size (tuple): Target size (width, height).
            original_size (tuple): Original size (width, height).
        Returns:
            resized_image (array): Resized image.
            resized_bboxes (array): Resized bounding boxes.
        """
        resized_image = cv2.resize(image, size)
        sw, sh = size[0] / original_size[0], size[1] / original_size[1]
        resized_bboxes = bboxes.copy()
        resized_bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * sw
        resized_bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * sh
        return resized_image, resized_bboxes
