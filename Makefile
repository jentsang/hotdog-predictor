# Makefile for group 17



#downloading the data 
data/raw/food_vendors_raw.csv : scripts/download_data.py
	python scripts/download_data.py \
		--source 'https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/food-vendors/exports/csv?lang=en&timezone=America%2FLos_Angeles&use_labels=true&delimiter=%3B' \
		--out-file data/raw/food_vendors_raw.csv

#preparing/splitting the data 
data/processed/vendors_train.csv data/processed/vendors_test.csv : data/raw/food_vendors_raw.csv scripts/prepare_data.py
	python scripts/prepare_data.py \
		 --input-file data/raw/food_vendors_raw.csv \
 		 --train-out data/processed/vendors_train.csv \
 		 --test-out data/processed/vendors_test.csv \
 		 --train-size 0.7 \
 		 --seed 522

#exploratory data analysis (EDA)
results/figures/EDA/plot1_cuisine_types.png results/figures/EDA/plot2_class_imbalance.png results/figures/EDA/plot3_blank_names_vs_hotdog.png : data/processed/vendors_train.csv scripts/eda.py
	python scripts/eda.py \
		--training-data data/processed/vendors_train.csv \
		--plot-to results/figures/EDA

#dummy baseline model
results/tables/Dummy/agg_cv_scores.csv results/tables/Dummy/raw_cv_scores.csv : data/processed/vendors_train.csv  data/processed/vendors_test.csv scripts/dummy_analysis.py
	python scripts/dummy_analysis.py \
		--training-data data/processed/vendors_train.csv \
		--testing-data data/processed/vendors_test.csv \
		--tables-to results/tables/Dummy \
		--seed 522 



