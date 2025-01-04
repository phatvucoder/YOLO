# Dataset Configuration
DATASET_NAME = "VOC"  # Name of the dataset: VOC or COCO
VOC_ROOT = "./VOCdevkit/VOC2007"  # Path to VOC data
COCO_ROOT = "./COCO"  # Path to COCO data
COCO_ANNOTATION_FILE = "./COCO/annotations/instances_train2017.json"  # Annotation file for COCO

# Model Configuration
NUM_CLASSES = 20  # Number of object classes (20 for VOC, 80 for COCO)
ANCHORS = [
    [97.64, 153.26], 
    [145.7, 301.49], 
    [341.09, 352.0], 
    [291.44, 186.93], 
    [39.53, 58.77]
]  # Anchor boxes (width, height) - Adjust according to the dataset
INPUT_SIZE = 416  # Input size (416x416)

# Training Configuration
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MOMENTUM = 0.9  # Momentum for SGD
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 50
SAVE_DIR = "./checkpoints"  # Directory to save checkpoints

# Loss Configuration
LAMBDA_COORD = 5  # Weight for coordinate loss
LAMBDA_NOOBJ = 0.5  # Weight for no-object loss

# Evaluation Configuration
CONF_THRESHOLD = 0.5  # Confidence threshold to retain bounding boxes
IOU_THRESHOLD = 0.5  # IoU threshold for nms
