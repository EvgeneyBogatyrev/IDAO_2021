import matplotlib.pyplot as plt

from reader import Reader


r = Reader("train/ER")

for i in range(10):
    image, part_type, energy = r.next()
    plt.imshow(image)
    print(part_type, energy)
    plt.show()
