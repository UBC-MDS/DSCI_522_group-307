# Income Level Predictor
A data analysis project for DSCI 522 Data Science Workflows

Authors: Saurav Chowdhury, Evhen Dytyniak, and Reiko Okamoto </br>
Date: 2020-01-24

## About

## Report
The report can be found [here]().

## Usage
To replicate this analysis, clone this repository, install the dependencies, and run the following commands at the command line from the root directory. 

```
# Download data

# Pre-process data

# Create EDA tables and figures
Rscript scripts/3_eda.R --train=clean_train_data.feather --out_dir=data --out_dir_plt=imgs

# Run ML analysis

# Render final report
```

## Dependencies
- Python 3.7.3 and Python packages:
    - docopt == 
    - requests == 
    - os == 
    - pandas ==
    - numpy ==    
    - scikit-learn == 
    - feather == 
- R version 3.6.1 and R packages:
    - knitr == 1.27.2
    - feather == 0.3.5
    - tidyverse == 1.3.0
    - plyr == 1.8.4
    - docopt == 0.6.1
    - ggridges == 0.5.2
    - ggthemes == 4.2.0

## References