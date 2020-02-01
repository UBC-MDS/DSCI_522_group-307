# Author: Saurav Chowdhury
# Date: 2020-22-01
"""
This script takes the downloaded file and clean/process/transform the data for further analysis. It then saves it
in desired location based on whether it is a training data or a test data. In case of training data it splits it 
into train and validation sets. 

Usage: scripts/2_wrangle_data.py --in_file=<input_file> --out_dir=<out_dir> [--istrain=<istrain>]

Options:
--in_file=<in_file>    Input train file on which wrangling will be done.
--out_dir=<out_dir>    Output directory(folder name) relative to root where to write the clean output in .csv format
[--istrain=<istrain>]    Optional argument, indicator whether it is a training set which will have 2 output train and validation or a test set which will have 1 clean output only

"""
    
from docopt import docopt
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import feather

opt = docopt(__doc__)

def main(in_file, out_dir, istrain=1):
    
    #--------fetching column names
    url_names = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names'
    names  = pd.read_table(url_names, header = None)
  
    column_names = list()
    for i in names.iloc[93:107,0]:
        column_names.append(i.split(':')[0])
    column_names.append('target')
    
    df = pd.read_csv(in_file)
    
    #--------adding custom column names and formatting them
    df.columns = column_names
    df.columns = df.columns.str.replace('-', '_') 

    #--------categorizing numerical and categorical features
    numerical = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    categorical = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'target']
    
    #-------- getting rid of blanks at the start of categorical features
    df[categorical] = df[categorical].apply(lambda x: x.str.strip())
    
    #-------- removing missing values
    df = df[(df.native_country != '?') & (df.occupation != '?') & (df.workclass !='?')]
    
    #-------- changing other native_countries to non-US, since US is predominant
    df.loc[df['native_country'] != 'United-States', 'native_country'] = 'Non-US'
    
    cap_gain_98 = np.quantile(df['capital_gain'], 0.98)
    cap_loss_98 = np.quantile(df['capital_loss'], 0.98)
    
    #-------- removing outliers from capital_gain and capital_loss by capping method
    df.loc[df['capital_gain'] > cap_gain_98, 'capital_gain'] = cap_gain_98
    df.loc[df['capital_loss'] > cap_loss_98, 'capital_loss'] = cap_loss_98
    
   
    if (istrain is None) or (istrain == "1"):
      
      X = df.drop('target', axis = 1)
      y = df['target']
      X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, random_state = 2020)
      X_train['target'] = y_train
      X_valid['target'] = y_valid
      
      X_train.to_csv(out_dir+'/clean_train_data.csv', index=False)
      X_valid.to_csv(out_dir+'/clean_validation_data.csv', index=False)
      
      feather.write_dataframe(X_train, out_dir+'/clean_train_data.feather')
      feather.write_dataframe(X_valid, out_dir+'/clean_validation_data.feather')
      
    else:
      
      df.to_csv(out_dir+'/clean_test_data.csv', index=False)
      feather.write_dataframe(df, out_dir+'/clean_test_data.feather')
            
    return

def test_fun():
  """
  This functions checks if the main function is able to take input file wrangle it and store at a specific location
  
  """
  
  #----------- getting sample data with 10 rows--------------------#
  
  test_df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test', skiprows = 1, nrows = 10)
  test_df.to_csv("data/sample_wrangle_data.csv", index = False)
  
  in_file1 = "data/sample_wrangle_data.csv"
  out_dir1 = "data"
  
  #-------------- check for train data-----------------------#
  main(in_file1, out_dir1)
  assert os.path.exists(out_dir1+'/clean_train_data.csv'), "Clean train data not found in location"
  
  temp_df = pd.read_csv(out_dir1+'/clean_train_data.csv')

  assert temp_df.shape[0] == 24129, "Number of columns not expected"

  #-------------- check for test data-----------------------#
  main(in_file1, out_dir1, 3)
  assert os.path.exists(out_dir1+'/clean_test_data.csv'), "Clean test data not found in location"

  temp_df = pd.read_csv(out_dir1+'/clean_test_data.csv')

  assert temp_df.shape[0] == 8, "Number of rows not expected"

if __name__ == "__main__":
  main(opt["--in_file"], opt["--out_dir"], opt["--istrain"])    
  test_fun()
