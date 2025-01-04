import os
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.voc import VOCDataset
from datasets.coco import COCODataset
from models.yolov2 import YOLOv2
from utils.loss import YOLOv2Loss
import config

def encode_yolo_target(bboxes, anchors, S=13, B=5, num_classes=20, input_size=416):
    """
    Encode bounding boxes into YOLO format.

    Args:
        bboxes (Tensor[N, 5]): Bounding boxes for the image.
        anchors (list): List of anchor boxes.
        S (int): Grid size.
        B (int): Number of anchors per grid cell.
        num_classes (int): Number of object classes.
        input_size (int): Size of the input image.

    Returns:
        Tensor[S, S, B, 5 + num_classes]
    """
    target = torch.zeros((S, S, B, 5 + num_classes), dtype=torch.float32)
    cell_size = input_size / S

    for bbox in bboxes:
        xmin, ymin, xmax, ymax, cls_id = bbox
        cls_id = int(cls_id)
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        bw = xmax - xmin
        bh = ymax - ymin

        grid_x = int(cx // cell_size)
        grid_y = int(cy // cell_size)

        if grid_x < 0 or grid_x >= S or grid_y < 0 or grid_y >= S:
            continue

        x_cell = (cx / cell_size) - grid_x
        y_cell = (cy / cell_size) - grid_y

        # Determine the best anchor
        best_iou = 0
        best_anchor = 0
        for i, (anchor_w, anchor_h) in enumerate(anchors):
            inter_w = min(bw, anchor_w)
            inter_h = min(bh, anchor_h)
            inter_area = inter_w * inter_h
            union_area = bw * bh + anchor_w * anchor_h - inter_area
            iou = inter_area / (union_area + 1e-16)
            if iou > best_iou:
                best_iou = iou
                best_anchor = i

        target[grid_y, grid_x, best_anchor, 0] = x_cell
        target[grid_y, grid_x, best_anchor, 1] = y_cell
        target[grid_y, grid_x, best_anchor, 2] = bw
        target[grid_y, grid_x, best_anchor, 3] = bh
        target[grid_y, grid_x, best_anchor, 4] = 1.0
        target[grid_y, grid_x, best_anchor, 5 + cls_id] = 1.0

    return target

def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes.

    Args:
        batch (list): List of tuples (image, bboxes).

    Returns:
        images (Tensor): Batch of images [B, 3, 416, 416].
        targets (Tensor): Batch of targets [B, S, S, B, 5 + num_classes].
    """
    images, targets = zip(*batch)
    images = torch.stack([img for img in images], dim=0)  # [B, 3, 416, 416]
    encoded_targets = []
    for bboxes in targets:
        y_encoded = encode_yolo_target(
            bboxes,
            anchors=config.ANCHORS,
            S=13,
            B=len(config.ANCHORS),  # Number of anchors
            num_classes=config.NUM_CLASSES,
            input_size=config.INPUT_SIZE
        )
        encoded_targets.append(y_encoded)
    encoded_targets = torch.stack(encoded_targets, dim=0)  # [B, S, S, B, 5 + num_classes]
    return images, encoded_targets

def get_dataloader(dataset_name, root, annotation_file=None, batch_size=16, input_size=416):
    """
    Get DataLoader for the specified dataset.

    Args:
        dataset_name (str): "VOC" or "COCO".
        root (str): Root directory of the dataset.
        annotation_file (str, optional): Path to COCO annotation file.
        batch_size (int): Batch size.
        input_size (int): Input image size.

    Returns:
        DataLoader
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if dataset_name == "VOC":
        dataset = VOCDataset(root=root, image_set="trainval", input_size=input_size, transform=transform)
    elif dataset_name == "COCO":
        dataset = COCODataset(root=root, annotation_file=annotation_file, input_size=input_size, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): YOLOv2 model.
        dataloader (DataLoader): Training DataLoader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        device (torch.device): Device to train on.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0

    for images, targets in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)           # [B, 3, 416, 416]
        targets = targets.to(device)         # [B, S, S, B, 5 + num_classes]

        outputs = model(images)              # [B, S, S, B, 5 + num_classes]
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

def save_checkpoint(model, optimizer, epoch, save_path):
    """
    Save model checkpoint.

    Args:
        model (nn.Module): Model to save.
        optimizer (Optimizer): Optimizer state.
        epoch (int): Current epoch.
        save_path (str): Path to save the checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, save_path)

def main():
    # Configuration
    dataset_name = config.DATASET_NAME
    root = config.VOC_ROOT if dataset_name == "VOC" else config.COCO_ROOT
    annotation_file = config.COCO_ANNOTATION_FILE if dataset_name == "COCO" else None
    input_size = config.INPUT_SIZE
    batch_size = 1
    learning_rate = config.LEARNING_RATE
    num_epochs = config.NUM_EPOCHS
    save_dir = config.SAVE_DIR
    os.makedirs(save_dir, exist_ok=True)

    # DataLoader
    dataloader = get_dataloader(dataset_name, root, annotation_file, batch_size, input_size)

    # Model
    model = YOLOv2(num_classes=config.NUM_CLASSES, anchors=config.ANCHORS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and Optimizer
    criterion = YOLOv2Loss(S=13, B=len(config.ANCHORS), num_classes=config.NUM_CLASSES,
                          lambda_coord=config.LAMBDA_COORD, lambda_noobj=config.LAMBDA_NOOBJ)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")

        # Save checkpoint
        if epoch % 10 == 0:
            save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, save_path)
            print(f"Checkpoint saved at {save_path}")

    print("Training complete!")

if __name__ == "__main__":
    main()
