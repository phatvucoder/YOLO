# pascal_voc.py
import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class PascalVOCDataset(Dataset):
    def __init__(
        self,
        root_dir,
        image_set='trainval',
        year='2007',
        img_size=448,
        grid_size=7,
        num_boxes=2,
        num_classes=20,
        transform=None
    ):
        self.root_dir = root_dir
        self.image_set = image_set
        self.year = year
        self.img_size = img_size
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Default transform
        if transform is None:
            self.transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),  # Convert PIL -> tensor [0..1]
            ])
        else:
            self.transform = transform

        self.image_dir = os.path.join(root_dir, f"VOC{year}/JPEGImages")
        self.annotation_dir = os.path.join(root_dir, f"VOC{year}/Annotations")
        self.image_set_file = os.path.join(root_dir, f"VOC{year}/ImageSets/Main/{image_set}.txt")

        with open(self.image_set_file) as f:
            self.image_ids = [line.strip() for line in f]

        # class
        self.class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        # Load PIL
        image = Image.open(img_path).convert("RGB")

        # Load annotation
        boxes, labels = self._load_annotation(image_id)
        target = self._encode_target(boxes, labels)

        # image => tensor
        image = self.transform(image)  # (3, H, W)

        return image, target

    def _load_annotation(self, image_id):
        annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            cls_id = self.class_to_idx[cls_name]

            bbox = obj.find("bndbox")
            x_min = float(bbox.find("xmin").text)
            y_min = float(bbox.find("ymin").text)
            x_max = float(bbox.find("xmax").text)
            y_max = float(bbox.find("ymax").text)

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(cls_id)

        return boxes, labels

    def _encode_target(self, boxes, labels):
        S, B, C = self.grid_size, self.num_boxes, self.num_classes
        target = torch.zeros((S, S, 5*B + C))

        for box, label in zip(boxes, labels):
            x_min, y_min, x_max, y_max = box
            w_img = self.img_size
            h_img = self.img_size

            # Calc center, w,h (normalize [0..1], 448x448)
            x_center = (x_min + x_max)/2.0 / w_img
            y_center = (y_min + y_max)/2.0 / h_img
            width = (x_max - x_min)/w_img
            height = (y_max - y_min)/h_img

            # Xác định cell
            grid_x = int(x_center * S)
            grid_y = int(y_center * S)
            if grid_x >= S: grid_x = S - 1
            if grid_y >= S: grid_y = S - 1

            x_cell = x_center * S - grid_x
            y_cell = y_center * S - grid_y

            # box0 = box
            target[grid_y, grid_x, 0] = x_cell
            target[grid_y, grid_x, 1] = y_cell
            target[grid_y, grid_x, 2] = width
            target[grid_y, grid_x, 3] = height
            target[grid_y, grid_x, 4] = 1.0  # obj conf

            # One-hot class
            target[grid_y, grid_x, 5 + label] = 1.0

        return target
