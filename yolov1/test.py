import torch
from config import IMG_SIZE, GRID_SIZE
from models.yolov1 import YOLOv1
# from utils.nms import non_max_suppression

def main():
    # load model
    model = YOLOv1(gridSize=GRID_SIZE, numBoxes=2, numClasses=20)
    model.load_state_dict(torch.load("yolov1.pth", map_location="cpu"))
    model.eval()

    # prediction
    with torch.no_grad():
        images = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)  # batch = 1
        outputs = model(images)  # (N, S, S, 5B + C)

        print(outputs)

if __name__ == "__main__":
    main()
