# Author: DSCI 522 - Group 307
# Date: 2020-22-01
#
"""This script downloads all the required dataset from a url and store it as csv at a desired location. 

Usage: scripts/1_download_data.py --url=<url> --out_path=<out_path> [--skiprows=<skiprows>] [--header=<header>]

Options:
--url=<url>              URL from where to download the data
--out_path=<out_path>    Path (including filename) relative to root where to locally write the file in .csv format
[--skiprows=<skiprows>]  Optional argument to skip any metadata at the top of the input file
[--header=<header>]      Optional argument to mention the row number for column names in the file

"""

from docopt import docopt
import requests
import os
import pandas as pd

opt = docopt(__doc__)

def main(url, out_path, skiprows=None, header='infer'):
    
    #url_data = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    #url_test = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test'
 
    try: 
      request = requests.get(url)
      request.status_code == 200
    except Exception as req:
      print("Website at the provided url does not exist.")
      print(req)
    
    if skiprows is None:
      df = pd.read_csv(url, skiprows = skiprows, header = header)
    else:
      skiprows = int(skiprows)
      df = pd.read_csv(url, skiprows = skiprows, header = header)

    df.to_csv(out_path, index=False)

def test_fun():
  """
  This functions checks if the main function is able to download and store a file at a specific location
  
  """
  
  test_url = "http://mlr.cs.umass.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
  path = "data/auto_mpg.csv"
  main(test_url, path)
  assert os.path.exists(path), "File not found in location"
  

if __name__ == "__main__":
  test_fun()
  main(opt["--url"], opt["--out_path"], opt["--skiprows"], opt["--header"])
