import pandas as pd
from sklearn.model_selection import train_test_split

from reader import Reader
from features import *

def get_data():
    
    train_data = []

    paths = ['ER', 'NR']

    percents = 0
    for path in paths:

        reader = Reader("train/" + path)
        idx = 0
        while (reader):

            idx += 1
            if (idx == 3000):
                percents += 1
                print(25 * percents, "%")
                

            image, part_type, energy = reader.next()

            pixels, brightness, center = get_features(image)
            center_delta = (center[0] ** 2 + center[1] ** 2) ** (1/2)

            data = {}
            data.update({
                'nonzero pixels' : pixels,
                'mean brightness' : brightness,
                'center of mass' : center_delta,
                'type' : part_type,
                'energy' : energy        
            })

            train_data.append(data)
        percents += 1

    train_data_pd = pd.DataFrame(train_data)
    print(train_data_pd)
    train_data_pd.to_csv("spreadsheets/data.csv")
    return train_data_pd



