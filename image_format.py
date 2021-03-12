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
