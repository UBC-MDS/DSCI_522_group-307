all : doc/income_level_report.md

data/adult_train_data.csv : scripts/1_download_data.py 
	python scripts/1_download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data" --out_path="data/adult_train_data.csv"

data/adult_test_data.csv: scripts/1_download_data.py 
	python scripts/1_download_data.py --url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test" --out_path="data/adult_test_data.csv" --skiprows=1

data/clean_train_data.feather data/clean_validation_data.feather : scripts/2_wrangle_data.py data/adult_train_data.csv
	python scripts/2_wrangle_data.py --in_file="data/adult_train_data.csv" --out_dir="data"

data/clean_test_data.feather : scripts/2_wrangle_data.py data/adult_test_data.csv
	python scripts/2_wrangle_data.py --in_file="data/adult_test_data.csv" --out_dir="data" --istrain=0

results/num_feat_summary.csv results/cat_feat_summary.csv results/dist_num_feat.png : scripts/3_eda.R data/clean_train_data.feather
	Rscript scripts/3_eda.R --train=data/clean_train_data.feather --out_dir=results


results/grid_search_summary.csv results/neg_features.csv results/pos_features.csv results/sig_features.csv : scripts/4_ml_analysis.py data/clean_train_data.feather data/clean_validation_data.feather data/clean_test_data.feather
	python scripts/4_ml_analysis.py --train=data/clean_train_data.feather --valid=data/clean_validation_data.feather --test=data/clean_test_data.feather --outputdir=results

doc/income_level_report.md : doc/income_level_report.Rmd doc/income_level_predictor.bib results/dist_num_feat.png results/grid_search_summary.csv results/pos_features.csv results/neg_features.csv results/sig_features.csv
	Rscript -e "rmarkdown::render('doc/income_level_report.Rmd', output_format = 'github_document')"

clean:
	rm -rf data/*
	rm -rf doc/income_level_report.md
	rm -rf results/*

