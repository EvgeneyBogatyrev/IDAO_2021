import numpy as np

from image_format import *


def get_features(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = crop(image)
    image = blur(image)
    image = remove_back(image)

    return nonzero_pixels(image), mean_brightness(image), center_of_mass(image)


def nonzero_pixels(image):
    return len(image[image > 0].flatten())


def mean_brightness(image):
    return np.mean(image[image > 0])


def center_of_mass(image):
    center_x, center_y, amount = 0, 0, 0

    for x in range(len(image)):
        for y in range(len(image[x])):
            if (image[x][y] > 0):
                center_x += x
                center_y += y
                amount += 1
    
    if (amount == 0):
        return image.shape[0] / 2, image.shape[1] / 2

    center_x /= amount
    center_y /= amount

    return center_x, center_y 
