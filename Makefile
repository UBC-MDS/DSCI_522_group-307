# Income Level Predictor Data Pipe
# author: Evhen Dytyniak
# date: 2020-02-01

#Runs all of the scripts and dependencies (as required if dependancy files/scripts updated)
all : doc/income_level_report.md

#Downloads training data from UCI and writes to a .csv
data/adult_train_data.csv : scripts/1_download_data.py 
	python scripts/1_download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" --out_path="data/adult_train_data.csv"

#Downloads test data from UCI and writes to a .csv
data/adult_test_data.csv: scripts/1_download_data.py 
	python scripts/1_download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" --out_path="data/adult_test_data.csv" --skiprows=1

#Wrangles training data and splits into train and validation sets then writes to .csv and .feather files
data/clean_train_data.feather data/clean_validation_data.feather : scripts/2_wrangle_data.py data/adult_train_data.csv
	python scripts/2_wrangle_data.py --in_file="data/adult_train_data.csv" --out_dir="data"

#Wrangles test data then writes to .csv and .feather files
data/clean_test_data.feather : scripts/2_wrangle_data.py data/adult_test_data.csv
	python scripts/2_wrangle_data.py --in_file="data/adult_test_data.csv" --out_dir="data" --istrain=0

#Exploratory data analysis - generates a numerical feature summary .png
results/num_feat_summary.csv results/cat_feat_summary.csv results/numerical.png : scripts/3_eda.R data/clean_train_data.feather
	Rscript scripts/3_eda.R --train=data/clean_train_data.feather --out_dir=results

#Data analysis using machine learning techniques to identify model performance and feature importance
#outputs include .csv of model scores and most/least important features
results/grid_search_summary.csv results/neg_features.csv results/pos_features.csv results/sig_features.csv : scripts/4_ml_analysis.py data/clean_train_data.feather data/clean_validation_data.feather data/clean_test_data.feather
	python scripts/4_ml_analysis.py --train=data/clean_train_data.feather --valid=data/clean_validation_data.feather --test=data/clean_test_data.feather --outputdir=results

#Generation of final report (.html optimum viewing format)
doc/income_level_report.md : doc/income_level_report.Rmd doc/income_level_predictor.bib results/numerical.png results/grid_search_summary.csv results/pos_features.csv results/neg_features.csv results/sig_features.csv
	Rscript -e "library(rmarkdown); render('doc/income_level_report.Rmd', output_format = 'github_document')"

#Removes all generated outputs from the data pipe
clean:
	rm -rf data/*
	rm -rf doc/income_level_report.md
	rm -rf results/*

#For milestone 3 purposes, only removes files that do not trigger the machine learning script (running this script takes 30+ min)
clean_light:  
	rm -rf doc/income_level_report.md
	rm -rf results/numerical.png