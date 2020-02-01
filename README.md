# Income Level Predictor
A data analysis project for DSCI 522 Data Science Workflows

Authors: Saurav Chowdhury, Evhen Dytyniak, and Reiko Okamoto 
Date: 2020-01-24

## About

This analysis attempted determine the most important features when predicting a yearly salary of more than 50,000 USD. A logistic regression model and AdaBoost model were trained in an effort to extract feature importance. The models did not perform exceedingly well, with scores in the low 80s, but performed similarly to random forest and support vector machine (SVM) classifiers. The logistic regression model’s most important features in predicting a yearly salary of greater than 50,000 USD were `marital_status_Married-AF-spouse` and `marital_status_Married-civ-spouse` while the most important features in predicting a yearly salary of less than 50,000 USD were `occupation_Priv-house-serv` and `workclass_Without-pay`. The Adaboost model identified `education_num` and `age` as the most important features in classification.

The data used in this project was created by Ronny Kohavi and Barry Becker, Data Mining and Visualization division at Silicon Graphics. This data was extracted from the 1994 US Census Data. It was sourced from the UCI Machine Learning Repository (Dua and Graff 2017) and can be found [here](https://archive.ics.uci.edu/ml/datasets/adult). Each row in the data represents the attributes of an individual such as sex, race, age, educational attainment, and working hours. The target variable is whether one's income is above or below 50,000 USD.  
## Report
The report can be found [here](https://github.com/UBC-MDS/DSCI_522_group-307/blob/master/doc/income_level_report.md).

## Usage
To replicate this analysis, clone this repository, install the dependencies, and run the following at the command line from the root directory. **_NOTE: the ML analysis can take up to an hour to run._**

```
make all
```
To start over with a clean slate with no intermediate or final outputs, run the following at the command line from the root directory:

```
make clean
```

Alternatively, to just remove the output of the EDA and report, run the following at the command line from the root directory. This essentially prevents the user from re-running the ML analysis that can take up to an hour to run.

```
make clean_light
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
