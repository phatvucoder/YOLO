import os
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import config
from datasets.voc import VOCDataset
from datasets.coco import COCODataset
from models.yolov3 import YOLOv3
from utils.loss import YOLOv3Loss
import math

def encode_yolo_target(bboxes, image_size, grid_size, anchors, stride):
    t = torch.zeros((grid_size, grid_size, len(anchors), 5 + config.NUM_CLASSES))
    scale = image_size / stride
    for bbox in bboxes:
        gx = (bbox[0] + bbox[2]) / 2.0
        gy = (bbox[1] + bbox[3]) / 2.0
        gw = bbox[2] - bbox[0]
        gh = bbox[3] - bbox[1]
        cx = gx * (scale / image_size)
        cy = gy * (scale / image_size)
        gw_s = gw * (scale / image_size)
        gh_s = gh * (scale / image_size)
        gi = int(cx)
        gj = int(cy)
        if gi < grid_size and gj < grid_size:
            best_iou = 0
            best_anchor = 0
            for i, anchor in enumerate(anchors):
                aw = anchor[0] / stride
                ah = anchor[1] / stride
                inter = min(gw_s, aw) * min(gh_s, ah)
                union = gw_s * gh_s + aw * ah - inter
                iou = inter / union
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = i
            tx = cx - gi
            ty = cy - gj
            tw = math.log(gw_s / (anchors[best_anchor][0] / stride) + 1e-16)
            th = math.log(gh_s / (anchors[best_anchor][1] / stride) + 1e-16)
            t[gj, gi, best_anchor, 0] = tx
            t[gj, gi, best_anchor, 1] = ty
            t[gj, gi, best_anchor, 2] = tw
            t[gj, gi, best_anchor, 3] = th
            t[gj, gi, best_anchor, 4] = 1.0
            cls_id = int(bbox[4])
            t[gj, gi, best_anchor, 5 + cls_id] = 1.0
    return t

def collate_fn(batch):
    images, all_bboxes = zip(*batch)
    images = torch.stack(images, dim=0)
    s1 = config.INPUT_SIZE // 32
    s2 = config.INPUT_SIZE // 16
    s3 = config.INPUT_SIZE // 8
    t1, t2, t3 = [], [], []
    for bboxes in all_bboxes:
        t1.append(encode_yolo_target(bboxes, config.INPUT_SIZE, s1, config.ANCHORS[0], 32))
        t2.append(encode_yolo_target(bboxes, config.INPUT_SIZE, s2, config.ANCHORS[1], 16))
        t3.append(encode_yolo_target(bboxes, config.INPUT_SIZE, s3, config.ANCHORS[2], 8))
    t1 = torch.stack(t1, dim=0)
    t2 = torch.stack(t2, dim=0)
    t3 = torch.stack(t3, dim=0)
    return images, (t1, t2, t3)

def get_dataloader():
    transform = transforms.Compose([transforms.ToTensor()])
    if config.DATASET_NAME == "VOC":
        dataset = VOCDataset(root=config.VOC_ROOT, image_set="trainval", input_size=config.INPUT_SIZE, transform=transform)
    elif config.DATASET_NAME == "COCO":
        dataset = COCODataset(root=config.COCO_ROOT, annotation_file=config.COCO_ANNOTATION_FILE, input_size=config.INPUT_SIZE, transform=transform)
    else:
        raise ValueError("Unsupported dataset.")
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    return dataloader

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    yolo_loss_13 = YOLOv3Loss(num_classes=config.NUM_CLASSES, anchors=config.ANCHORS[0], stride=32, lambda_coord=config.LAMBDA_COORD, lambda_noobj=config.LAMBDA_NOOBJ)
    yolo_loss_26 = YOLOv3Loss(num_classes=config.NUM_CLASSES, anchors=config.ANCHORS[1], stride=16, lambda_coord=config.LAMBDA_COORD, lambda_noobj=config.LAMBDA_NOOBJ)
    yolo_loss_52 = YOLOv3Loss(num_classes=config.NUM_CLASSES, anchors=config.ANCHORS[2], stride=8, lambda_coord=config.LAMBDA_COORD, lambda_noobj=config.LAMBDA_NOOBJ)
    total_loss_epoch = 0.0
    for images, (targets_13, targets_26, targets_52) in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        targets_13 = targets_13.to(device)
        targets_26 = targets_26.to(device)
        targets_52 = targets_52.to(device)
        out_13, out_26, out_52 = model(images)
        loss_13 = yolo_loss_13(out_13, targets_13, input_size=config.INPUT_SIZE)
        loss_26 = yolo_loss_26(out_26, targets_26, input_size=config.INPUT_SIZE)
        loss_52 = yolo_loss_52(out_52, targets_52, input_size=config.INPUT_SIZE)
        loss = loss_13 + loss_26 + loss_52
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_epoch += loss.item()
    return total_loss_epoch / len(dataloader)

def save_checkpoint(model, optimizer, epoch, save_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, save_path)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_dataloader()
    model = YOLOv3(num_classes=config.NUM_CLASSES, anchors=config.ANCHORS).to(device)
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"Epoch {epoch}/{config.NUM_EPOCHS}")
        train_loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Loss: {train_loss:.4f}")
        if epoch % 10 == 0:
            save_path = os.path.join(config.SAVE_DIR, f"yolov3_epoch_{epoch}.pth")
            save_checkpoint(model, optimizer, epoch, save_path)
            print(f"Checkpoint saved: {save_path}")
    print("Done.")

if __name__ == "__main__":
    main()
