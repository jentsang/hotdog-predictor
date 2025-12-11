#mock run all script to see if it works similr to lectyre slide. will delete after 

# ----------------------------------------
# 1. Download the data
# ----------------------------------------
python scripts/download_data.py \
  --source "https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/food-vendors/exports/csv?lang=en&timezone=America%2FLos_Angeles&use_labels=true&delimiter=%3B" \
  --out-file data/raw/food_vendors_raw.csv


# ----------------------------------------
# 2. Prepare / split the data
# ----------------------------------------
python scripts/prepare_data.py \
  --input-file data/raw/food_vendors_raw.csv \
  --train-out data/processed/vendors_train.csv \
  --test-out data/processed/vendors_test.csv \
  --train-size 0.7 \
  --seed 522


# ----------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ----------------------------------------
python scripts/eda.py \
  --training-data data/processed/vendors_train.csv \
  --plot-to results/figures/EDA


# ----------------------------------------
# 4. Dummy Baseline Model
# ----------------------------------------
python scripts/dummy_analysis.py \
  --training-data data/processed/vendors_train.csv \
  --testing-data data/processed/vendors_test.csv \
  --tables-to results/tables/ \
  --seed 522


# ----------------------------------------
# 5. Decision Tree Model
# ----------------------------------------
python scripts/decisiontree_analysis.py \
  --training-data data/processed/vendors_train.csv \
  --testing-data data/processed/vendors_test.csv \
  --figures-to results/figures/ \
  --tables-to results/tables/ \
  --model-to results/models/ \
  --seed 522


# ----------------------------------------
# 6. Logistic Regression Model
# ----------------------------------------
python scripts/logistic_regression_analysis.py \
  --training-data data/processed/vendors_train.csv \
  --testing-data data/processed/vendors_test.csv \
  --figures-to results/figures/ \
  --tables-to results/tables/ \
  --model-to results/models/ \
  --seed 522


# ----------------------------------------
# 7. Bayesian Model
# ----------------------------------------
python scripts/bayesian_analysis.py \
  --training-data data/processed/vendors_train.csv \
  --testing-data data/processed/vendors_test.csv \
  --figures-to results/figures/ \
  --tables-to results/tables/ \
  --model-to results/models/ \
  --seed 522


# ----------------------------------------
# 8. Compare Models
# ----------------------------------------
python scripts/compare_models.py \
  --table-dir=results/tables \
  --output-dir=results/tables \
  --param=mean


# ----------------------------------------
# 9. Bayesian Evaluation
# ----------------------------------------
python scripts/bayesian_evaluation.py \
  --training-data data/processed/vendors_train.csv \
  --testing-data data/processed/vendors_test.csv \
  --figures-to results/figures/ \
  --tables-to results/tables/ \
  --model-to results/models/ \
  --seed 522

