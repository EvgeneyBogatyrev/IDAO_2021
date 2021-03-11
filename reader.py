import os
import cv2

class Reader:
    def __init__(self, path, test=False):
        # path - путь к директории, test - тестовые или тренировочные данные
        self.path = path
        self.test = test
        
        self.images = os.listdir(path)
        self.images.sort()
        self.count = 0
        self.limit = len(self.images)


    def next(self):
        # Возвращает следующее изображение и его параметры (если test = False)
        if self.count >= self.limit:
            print("No more images")
            return

        image_path = os.path.join(self.path, self.images[self.count])
        image = cv2.imread(image_path)
        
        part_type = None
        energy = None
        if not self.test:
            part_type, energy = self.get_info(self.images[self.count])
        
        self.count += 1
        return image, part_type, energy


    def get_info(self, name):
        part_type = None
        energy = None
        
        symbols = ['.', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        index = 0
        while name[index] in symbols:
            index += 1

        string = "__CYGNO_60_40_"
        index += len(string)

        if name[index] == 'H':
            part_type = 0
            index += 6
        else:
            part_type = 1
            index += 3

        end = index
        while '0' <= name[end] <= '9':
            end += 1

        energy = ''.join(name[index : end])
        energy = int(energy)
        return part_type, energy
