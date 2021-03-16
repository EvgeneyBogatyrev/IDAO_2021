import numpy as np
import pandas as pd
import pickle


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

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


class Model:

    def __init__(self):
        self.data = pd.read_csv("spreadsheets/data.csv")

        self.part_type = self.data.pop("type")
        self.energy = self.data.pop("energy")


    def divide(self, data, labels):
        train, test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)
        return train, test, labels_train, labels_test


    def train_regressor(self, best_num_of_est=False):
        train, test, labels_train, labels_test = train_test_split(self.data, self.energy)

        self.regressor = GradientBoostingRegressor(
                criterion = "mae",
                max_depth = 3, 
                n_estimators = 113, 
                learning_rate = 0.2,
                min_samples_leaf = 2,
                min_samples_split = 3,
                subsample = 1
            )
        self.regressor.fit(train, labels_train)

        if (best_num_of_est):
            errors = [mean_absolute_error(labels_test, y_pred) for y_pred in self.regressor.staged_predict(test)]
            best_number_of_estimators = np.argmin(errors)

            print("Best number of estimators =", best_number_of_estimators)

            self.regressor = GradientBoostingRegressor(
                    criterion = "mae",
                    max_depth = 3, 
                    n_estimators = best_number_of_estimators, 
                    learning_rate = 0.2,
                    min_samples_leaf = 2,
                    min_samples_split = 3,
                    subsample = 0.8
                )

            self.regressor.fit(train, labels_train)


        y_pred = self.regressor.predict(test)

        y_pred = self.normalize_energy_values(y_pred)

        error = mean_absolute_error(y_pred, labels_test)
        print("MAE:", error)
        pickle.dump(self.regressor, open("regressor.pkl", "wb"))


    def grid_search_regressor(self, train, labels):
        param_grid = {
            'learning_rate' : [0.05, 0.1, 0.2],
            'n_estimators' : [80, 100, 120],
            'subsample' : [0.8, 1],
            'criterion' : ['mae'],
            'min_samples_split' : [2, 3, 5],
            'min_samples_leaf' : [1, 2, 3]
        }
        gbr = GradientBoostingRegressor()
        grid_search = GridSearchCV(estimator = gbr, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

        grid_search.fit(train, labels)
        print(grid_search.best_params_)

    
    def normalize_energy_values(self, values):
        possible_values = [1, 3, 6, 10, 20, 30]
        for i in range(len(values)):
            value = values[i] 

            min_error = -1
            best_value = -1
            
            for p_value in possible_values:
                error = abs(value - p_value)
                if (min_error == -1 or error < min_error):
                    min_error = error
                    best_value = p_value
       
            values[i] = best_value
        return values 
