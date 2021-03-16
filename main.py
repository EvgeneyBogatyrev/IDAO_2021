import matplotlib.pyplot as plt

from reader import Reader
from train import Model
from image_format import *
from features import *


def visualize():
    r = Reader("train/ER")
    skip = 200
    sample_size = 10

    for i in range(skip):
        _, _, _ = r.next()

    for i in range(sample_size):

        fig, axs = plt.subplots(2)

        image, part_type, energy = r.next()
    
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = crop(image)

        axs[0].imshow(image, cmap="gray")
    
        image = blur(image)
        image = remove_back(image)
        image = remove_pixels(image)

        axs[1].imshow(image, cmap="gray")
 
        print("№", i + 1 + skip, sep="")
        print("Type:", part_type) 
        print("Energy:", energy)
        print("Nonzero pixels number:", nonzero_pixels(image)) 
        print("Mean brightness:", mean_brightness(image))
        print("Center of mass:", center_of_mass(image))
        print("")

        plt.show()


model = Model()
model.train_regressor()
