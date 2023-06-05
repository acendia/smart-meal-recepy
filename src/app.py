#!/usr/bin/python3

# read image
import cv2
import numpy as np
import os
import sys
from ultralytics import YOLO


def read_image(image_path):
    """Read image from path

    Args:
        image_path (str): path to image

    Returns:
        np.array: image array
    """
    if not os.path.exists(image_path):
        print("Image path does not exist")
        sys.exit(1)

    image = cv2.imread(image_path)
    return image


# yolov8
def yolov8(image_path):
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    results = model(image_path)
    print(results)


def main():
    """Main function
    """
    image_path = "./imgs/broccoli.jpeg"
    # image = read_image(image_path)
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    yolov8(image_path)


if __name__ == "__main__":
    main()