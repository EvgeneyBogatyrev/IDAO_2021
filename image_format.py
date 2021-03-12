import cv2
import numpy as np

def image_format(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = image_gray.copy().astype(float)
    for x in range(1, len(image_blur) - 1):
        for y in range(1, len(image_blur[x]) - 1):
            block = image_blur[x - 1 : x + 1, y - 1 : y + 1]
            image_blur[x][y] = np.sum(block) / 9
    return image_blur

