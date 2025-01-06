import os

DATASET_NAME = "VOC"
VOC_ROOT = "./VOCdevkit/VOC2007"
COCO_ROOT = "./COCO"
COCO_ANNOTATION_FILE = "./COCO/annotations/instances_train2017.json"
NUM_CLASSES = 20
ANCHORS = [
    [[116, 90], [156, 198], [373, 326]], # for scale 13
    [[30, 61], [62, 45], [59, 119]],    # for scale 26
    [[10, 13], [16, 30], [33, 23]]     # for scale 52
]
INPUT_SIZE = 640
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 50
SAVE_DIR = "./checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5
CONF_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.5
