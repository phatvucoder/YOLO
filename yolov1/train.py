import torch
from torch.utils.data import DataLoader

from config import (
    IMG_SIZE,
    GRID_SIZE,
    BOXES_PER_CELL,
    NUM_CLASSES,
    LEARNING_RATE,
    EPOCHS,
    BATCH_SIZE,
    MOMENTUM,
    WEIGHT_DECAY,
    LAMBDA_COORD,
    LAMBDA_NOOBJ
)
from models.yolov1 import YOLOv1
from utils.loss import YOLOLoss
from datasets.pascal_voc import PascalVOCDataset
from utils.train_utils import train_one_epoch

def main():
    # path to VOC dataset
    root_img_dir = "../VOCdevkit"
    train_dataset = PascalVOCDataset(
        root_dir=root_img_dir,
        image_set='trainval',
        year='2007',
        img_size=IMG_SIZE,
        grid_size=GRID_SIZE,
        num_boxes=BOXES_PER_CELL,
        num_classes=NUM_CLASSES
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Init Model, Loss, Optimizer
    model = YOLOv1(gridSize=GRID_SIZE, numBoxes=BOXES_PER_CELL, numClasses=NUM_CLASSES)
    criterion = YOLOLoss(
        gridSize=GRID_SIZE,
        numBoxes=BOXES_PER_CELL,
        numClasses=NUM_CLASSES,
        lambda_coord=LAMBDA_COORD,
        lambda_noobj=LAMBDA_NOOBJ
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # Choose Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f'Device detected, Start training using {device}')
    model = model.to(device)
    criterion = criterion.to(device)

    # Training
    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, epoch, device=device)

    # Save model
    torch.save(model.state_dict(), "yolov1.pth")
    print("Finished training. Model saved as yolov1.pth")

if __name__ == "__main__":
    main()
