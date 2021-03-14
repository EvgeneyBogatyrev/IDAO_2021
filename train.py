import pandas as pd

from reader import Reader
from features import *

def get_data():
    
    train_data = []

    paths = ['ER', 'NR']

    for path in paths:

        reader = Reader("train/" + path)
        
        while (reader):
            image, part_type, energy = reader.next()

            pixels, brightness, center = get_features(image)

            data = {}
            data.update({
                'nonzero pixels' : pixels,
                'mean brightness' : brightness,
                'center of mass' : center,
                'type' : part_type,
                'energy' : energy        
            })

            train_data.append(data)

    train_data_pd = pd.DataFrame(train_data)
    print(train_data_pd)
    return train_data_pd

