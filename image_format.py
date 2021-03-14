import cv2
import numpy as np

def crop(image):
    return image[192 : 384, 192 : 384]

def blur(image):
    blurred_image = cv2.medianBlur(image, ksize=3)
    return blurred_image

def remove_back(image):
    image[np.where(image < 105)] = 0
    return image

def remove_pixels(image):
    image_copy = image.copy()
    for x in range(len(image)):
        if (x == 0 or x == len(image) - 1):
            continue
        for y in range(len(image[x])):
            if (y == 0 or y == len(image) - 1):
                continue
            block = image[x - 1 : x + 1, y - 1 : y + 1]
            if (np.sum(block)) < 3:
                image_copy[x][y] = 0
    return image_copy
