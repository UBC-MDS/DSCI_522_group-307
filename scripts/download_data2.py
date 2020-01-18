# Author: DSCI 522 - Group 307
# Date: 2020-18-01
#
# This script downloads all the required datasets and files and store them in desired location.
# This script takes no arguments.
#
# Usage: Python download_data.py


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def download_split_data():
    url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
    url_names = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'
    
    data = pd.read_csv(url_data, header = None)
    X_test = pd.read_csv(url_test, skiprows = 1, header = None)
    names  = pd.read_table(url_names, header = None)
  
    column_names = list()
    for i in names.iloc[93:107,0]:
        column_names.append(i.split(':')[0])
    column_names.append('target')

    data.columns = column_names
    X_test.columns = column_names
    
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, random_state = 2020)

    X_train.insert(loc = len(X_train.columns), column = 'target', value = y_train)
    X_valid.insert(loc = len(X_valid.columns), column = 'target', value = y_valid)

    X_train.to_csv('../data/train.csv')
    X_valid.to_csv('../data/validation.csv')
    X_test.to_csv('../data/test.csv')

    return


download_split_data()
