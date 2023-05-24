import torch
import fileinput
from utils_.colors import new_line

if __name__ == "__main__":
    torch.hub.load("ultralytics/yolov5", "yolov5s")
    torch.hub.load("ultralytics/yolov5", "yolov5l")

    # Modify the color for the bounding boxes
    with fileinput.FileInput('./fastapi/yolov5/utils/plots.py', inplace=True) as archivo:
        [print(new_line if 'self.palette =' in line else line, end='') for line in archivo]
