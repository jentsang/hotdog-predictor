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
		--tables-to results/tables\
		--seed 522 

#Decision Tree model 
results/tables/DecisionTree/agg_cv_scores.csv \
results/tables/DecisionTree/raw_cv_scores.csv \
results/tables/DecisionTree/train__model_mismatches.csv \
results/figures/DecisionTree/diagram.png \
results/figures/DecisionTree/train_confusion_matrix.png \
results/models/DecisionTree/DecisionTree.pickle: \
    data/processed/vendors_train.csv \
    data/processed/vendors_test.csv \
    scripts/decisiontree_analysis.py
	python scripts/decisiontree_analysis.py \
		--training-data data/processed/vendors_train.csv \
		--testing-data data/processed/vendors_test.csv \
		--figures-to results/figures \
		--tables-to results/tables \
		--model-to results/models \
		--seed 522

#Logistic Regression model
results/tables/LogisticRegression/agg_cv_scores.csv \
results/tables/LogisticRegression/raw_cv_scores.csv \
results/tables/LogisticRegression/coefficients.csv \
results/tables/LogisticRegression/train__model_mismatches.csv \
results/figures/LogisticRegression/most_discriminant_features.png \
results/figures/LogisticRegression/train_confusion_matrix.png \
results/models/LogisticRegression/LogisticRegression.pickle: \
    data/processed/vendors_train.csv \
    data/processed/vendors_test.csv \
    scripts/logistic_regression_analysis.py
	python scripts/logistic_regression_analysis.py \
		--training-data data/processed/vendors_train.csv \
		--testing-data data/processed/vendors_test.csv \
		--figures-to results/figures \
		--tables-to results/tables \
		--model-to results/models \
		--seed 522

#Bayesian model 
results/tables/NaiveBayes/raw_cv_scores.csv \
results/tables/NaiveBayes/agg_cv_scores.csv \
results/tables/NaiveBayes/train__model_mismatches.csv \
results/tables/NaiveBayes/RandomizedSearchCV_results_head.csv \
results/tables/NaiveBayes/best_rcv_model_test_mismatches.csv \
results/figures/NaiveBayes/train_confusion_matrix.png \
results/figures/NaiveBayes/best_rcv_model_test_confusion_matrix.png \
results/models/NaiveBayes/best_NaiveBayes.pickle: \
    data/processed/vendors_train.csv \
    data/processed/vendors_test.csv \
    scripts/bayesian_analysis.py
	python scripts/bayesian_analysis.py \
		--training-data data/processed/vendors_train.csv \
		--testing-data data/processed/vendors_test.csv \
		--figures-to results/figures \
		--tables-to results/tables \
		--model-to results/models \
		--seed 522

