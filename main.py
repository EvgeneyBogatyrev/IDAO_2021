import matplotlib.pyplot as plt

from reader import Reader
from image_format import *

r = Reader("train/ER")

for i in range(100):
    _, _, _ = r.next()
for i in range(10):
    image, part_type, energy = r.next()
    image = image[192 : 384, 192 : 384, :]
    image = image_format(image)
    plt.imshow(image)
    print(part_type, energy)
    plt.show()
