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
