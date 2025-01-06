import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import cv2
import numpy as np

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class VOCDataset(Dataset):
    def __init__(self, root, image_set, input_size=416, transform=None):
        self.root = root  # VOC data root path
        self.image_set = image_set  # trainval or test
        self.input_size = input_size  # Input image size
        self.transform = transform  # Image transform

        # VOC dataset classes
        self.image_dir = os.path.join(root, "JPEGImages")
        self.annotation_dir = os.path.join(root, "Annotations")
        with open(os.path.join(root, "ImageSets/Main", f"{image_set}.txt")) as f:
            self.image_list = f.read().strip().split("\n")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        """
        Get image and annotations.
        Args:
            index (int): Index of image.
        Returns:
            image (Tensor): Image. [C, H, W]
            bboxes (Tensor): Bounding boxes and labels. [N, 5]
        """
        image_id = self.image_list[index]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")

        # Read image in BGR format
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read annotations (bounding boxes)
        bboxes = self.parse_voc_annotation(annotation_path)

        # Resize image and bounding boxes
        h, w, _ = image.shape
        image, bboxes = self.resize(image, bboxes, (self.input_size, self.input_size), (w, h))

        # Transform image
        if self.transform:
            image = self.transform(image)

        return image, bboxes

    def parse_voc_annotation(self, annotation_path):
        """
        Parse VOC annotation.
        Args:
            annotation_path (str): Annotation file path.
        Returns:
            bboxes (array): Bounding boxes and labels. [xmin, ymin, xmax, ymax, label]
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        bboxes = []

        for obj in root.findall("object"):
            label = obj.find("name").text
            class_id = VOC_CLASSES.index(label)
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            bboxes.append([xmin, ymin, xmax, ymax, class_id])

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
            resized_image (array): Resized image. [H, W, C]
            resized_bboxes (array): Resized bounding boxes. [N, 5]
        """
        resized_image = cv2.resize(image, size)
        sw, sh = size[0] / original_size[0], size[1] / original_size[1]
        resized_bboxes = bboxes.copy()
        resized_bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * sw
        resized_bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * sh
        return resized_image, resized_bboxes
