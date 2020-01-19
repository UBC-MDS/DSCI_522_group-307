# Author: Saurav Chowdhury
# Date: 2020-16-01
#
# This script downloads all the required datasets and files and store them in desired location.
# This script takes no arguments.
#
# Usage: Python download_data.py

import urllib.request

print('Beginning file download with urllib2...')

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
urllib.request.urlretrieve(url, '../data/adult_train.csv')

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
urllib.request.urlretrieve(url, '../data/adult_test.csv')

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'
urllib.request.urlretrieve(url, '../data/adult_info.txt')