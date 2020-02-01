# Income Level Predictor
A data analysis project for DSCI 522 Data Science Workflows

Authors: Saurav Chowdhury, Evhen Dytyniak, and Reiko Okamoto </br>
Date: 2020-01-24

## About

This analysis attempted determine the most important features when predicting a yearly salary of more than 50,000 USD. A logistic regression model and AdaBoost model were trained in an effort to extract feature importance. The models did not perform exceedingly well, with scores in the low 80s, but performed similarly to random forest and support vector machine (SVM) classifiers. The logistic regression model’s most important features in predicting a yearly salary of greater than 50,000 USD were `marital_status_Married-AF-spouse` and `marital_status_Married-civ-spouse` while the most important features in predicting a yearly salary of less than 50,000 USD were `occupation_Priv-house-serv` and `workclass_Without-pay`. The Adaboost model identified `education_num` and `age` as the most important features in classification.

The data used in this project was created by Ronny Kohavi and Barry Becker, Data Mining and Visualization division at Silicon Graphics. This data was extracted from the 1994 US Census Data. It was sourced from the UCI Machine Learning Repository (Dua and Graff 2017) and can be found [here](https://archive.ics.uci.edu/ml/datasets/adult). Each row in the data represents the attributes of an individual such as sex, race, age, educational attainment, and working hours. The target variable is whether one's income is above or below 50,000 USD.  
## Report
The report can be found [here](https://ubc-mds.github.io/DSCI_522_group-307/doc/income_level_report.html).

## Usage
To replicate this analysis, clone this repository, install the dependencies, and run the following commands at the command line from the root directory.  
</br>
__Note__  
- __Running `make clean` followed by `make all` will take up to an hour and consume all available processors__    
- __For the purpose of this milestone submission, `make clean_light` will only remove files that do not trigger the time-consuming scripts, but will demonstrate correct use of the Makefile__   
- __if using a Windows OS, it appears that downloading a local copy of the repo and running `make clean_light` followed by `make all`, runs all of the scripts in the pipeline (this is not the case for Linux or MacOS systems)__  
```
make clean
make all
```
OR, for this submission:
```
make clean_light
make all
```

## Dependencies
- Python 3.7.3 and Python packages:
    - docopt == 0.6.2
    - requests == 2.22.0
    - pandas == 0.25.3
    - numpy ==  1.18.1
    - scikit-learn == 0.22.1
    - feather-format == 0.4.0
    - pyarrow == 0.15.1
- R version 3.6.1 and R packages:
    - knitr == 1.27.2
    - feather == 0.3.5
    - tidyverse == 1.3.0
    - docopt == 0.6.1
    - ggthemes == 4.2.0
    - testthat == 2.3.1
    - gridExtra == 2.3
    - rlang == 0.4.4
    - rmarkdown == 2.1
    
### License

The Income Level Predictor materials are licensed under the MIT License.

## References

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository . Irvine, CA: University of California, School of Information and Computer Science. [http://archive.ics.uci.edu/ml]
