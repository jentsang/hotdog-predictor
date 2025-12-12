# Food Vendors in Vancouver: Hot Dog or Not?

author: TSANG JENNIFER, TSEGAY SAMRAWIT MEZGEBO, ASLAM ZAKI, PALAFOX PRIETO HECTOR  

Demo of a data analysis project for DSCI 522 (Data Science Workflows), a course in the Master of Data Science program at the University of British Columbia.

## About

In this project, we use the City of Vancouver **“Food Vendors”** open dataset to explore where different kinds of food vendors operate in Vancouver and whether we can automatically identify hot-dog vendors.

Our main research question is:

> Can we predict whether a food vendor is a hot-dog vendor or not?

We construct a binary target variable `is_hotdog` using the `DESCRIPTION` column: it is `True` when `DESCRIPTION` is `"Hot Dogs"` and `False` otherwise. Our main feature is the vendor’s `BUSINESS_NAME`, which we use as text input to a set of classification models.

The goals of this project are to:

- build and compare simple binary classifiers on real open data,  
- work with text fields (business names) to construct a meaningful target variable,  
- follow a clear and reproducible workflow using scripts, Quarto, Docker, and GitHub.

We compare several models (Dummy baseline, Decision Tree, Logistic Regression, Naïve Bayes, and a Bayesian model) using cross-validation and a held-out test set. In our current results, a tuned Naïve Bayes classifier performs best, with test accuracy around 0.79 and relatively few false positives.

## Data

The data set used in this project is the **“Food Vendors”** dataset from the City of Vancouver Open Data Portal:

- **Dataset:** Food Vendors  
- **Provider:** City of Vancouver Open Data Portal  
- **URL:** https://opendata.vancouver.ca/explore/dataset/food-vendors/table/  

Each row in the dataset represents a food vendor and includes information such as the business name, food description, location, and geographic coordinates. For our analysis we primarily use:

- `BUSINESS_NAME` — the name of the vendor (feature)  
- `DESCRIPTION` — the category of food offered (used to create the `is_hotdog` target)

Other columns (e.g., `LOCATION`, `GEO_LOCALAREA`, coordinates) remain in the raw data but are not used directly in the final models.

The data is published under the **Open Government Licence – Vancouver**.  
For details, see: https://opendata.vancouver.ca/pages/licence/


## Report

The final report for this milestone is available in the `reports/` folder:

- HTML: [dog_or_not_report.html](reports/dog_or_not_report.html)  
- PDF:  [dog_or_not_report.pdf](reports/dog_or_not_report.pdf)


## Usage

### Working locally

From the root of this repository, the analysis can be run as follows (assuming `conda` is installed).

First time setup (recommended, using conda-lock.yml):

    git clone https://github.com/jentsang/DSCI-522-Group-17.git
    cd DSCI-522-Group-17

    # Create the environment from the lock file
    conda-lock install --name dsci522proj conda-lock.yml
    conda activate dsci522proj

Alternatively, you can create the environment directly from `environment.yml`:

    conda env create -f environment.yml
    conda activate dsci522proj

### Using Docker

If you want to run the docker image of this repo, you can instead just open `Docker Desktop`, and then in a terminal run:

```bash
docker pull jentsang/hotdog-predictor:sha-561dde5
```

And once the image is pulled, you can execute:

```bash
docker compose up
```

(Do not forget to close the container after finishing by using `Ctrl+C` and typing `docker compose rm` into the terminal)

To run the analysis you can just type:

```http://localhost:8888/```

In your browser (make sure there are no other instances of jupyterlab open)


Then, in JupyterLab, open `notebooks/dog_or_not.ipynb` and run the cells in order to reproduce the analysis.

Next, under the "Kernel" menu click "Restart Kernel and Run All Cells...".


## Running the scripts

The full workflow can be run from the command line using the scripts in the scripts/ directory.
From the root of the repository, run the following commands in order (either in your local conda environment or inside the Docker container):

Before anything, make sure you are in the root by performing:

```bash
pwd
```

If you are not in the root, you will need to change the relative directory of the paths presented below.

1. Download the data:

``` bash
python scripts/download_data.py \
  --source 'https://opendata.vancouver.ca/api/explore/v2.1/catalog/datasets/food-vendors/exports/csv?lang=en&timezone=America%2FLos_Angeles&use_labels=true&delimiter=%3B' \
  --out-file data/raw/food_vendors_raw.csv

```

2. Prepare/split the data

``` bash
python scripts/prepare_data.py \
  --input-file data/raw/food_vendors_raw.csv \
  --train-out data/processed/vendors_train.csv \
  --test-out data/processed/vendors_test.csv \
  --train-size 0.7 \
  --seed 522
```

3. Exploratory Data Analysis (EDA)

``` bash
python scripts/eda.py \
  --training-data data/processed/vendors_train.csv \
  --plot-to results/figures/EDA
```

4. Dummy Baseline Model

``` bash
python scripts/dummy_analysis.py \
  --training-data data/processed/vendors_train.csv \
  --testing-data data/processed/vendors_test.csv \
  --tables-to results/tables/ \
  --seed 522
```

5. Decision Tree Model

``` bash
python scripts/decisiontree_analysis.py \
  --training-data data/processed/vendors_train.csv \
  --testing-data data/processed/vendors_test.csv \
  --figures-to results/figures/ \
  --tables-to results/tables/ \
  --model-to results/models/ \
  --seed 522
```

6. Logistic Regression Model

``` bash
python scripts/logistic_regression_analysis.py \
  --training-data data/processed/vendors_train.csv \
  --testing-data data/processed/vendors_test.csv \
  --figures-to results/figures/ \
  --tables-to results/tables/ \
  --model-to results/models/ \
  --seed 522
```

7. Bayesian Model

``` bash
python scripts/bayesian_analysis.py \
  --training-data data/processed/vendors_train.csv \
  --testing-data data/processed/vendors_test.csv \
  --figures-to results/figures/ \
  --tables-to results/tables/ \
  --model-to results/models/ \
  --seed 522
```

8. Compare models

``` bash
python scripts/compare_models.py \
  --table-dir=results/tables \
  --output-dir=results/tables \
  --param=mean
```

9. Bayesian evaluation

``` bash
python scripts/bayesian_evaluation.py \
  --training-data data/processed/vendors_train.csv \
  --testing-data data/processed/vendors_test.csv \
  --figures-to results/figures/ \
  --tables-to results/tables/ \
  --model-to results/models/ \
  --seed 522
```

## Running tests

After setting up the environment, you can run the unit tests from the root of the repository.

To run the full test suite:

```bash
pytest tests/
```
To run a single test file, for example the tests for the Naive Bayes model
evaluation methods:

```bash
pytest tests/test_store_raw_cv_scores.py
pytest tests/test_store_agg_cv_scores.py
pytest tests/test_store_confusion_matrix.py
pytest tests/test_store_model_mismatches.py
```

## Dependencies

The main dependencies needed to run the analysis (as specified in environment.yml) are:

- python=3.11
- conda-lock=3.0.4
- scikit-learn=1.7.2
- pandas=2.3.3
- jupyterlab=4.4.10
- altair=6.0.0
- matplotlib=3.10.7
- pip=25.3
- pandera=0.26.1
- click=8.3.1
- vl-convert-python=1.8.0
- quarto=1.8.26
- pip:
    - mglearn==0.2.0

For the complete and authoritative list of packages and versions, please see environment.yml and conda-lock.yml.

## License

The written report and documentation contained in this repository are licensed under the Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license. See the LICENSE file for more information. If re-using or re-mixing the report, please provide attribution and a link to this repository.

The software code contained within this repository is licensed under the MIT License. See the LICENSE file for more information.

The Food Vendors dataset is provided by the City of Vancouver under the Open Government Licence – Vancouver. See the City of Vancouver open data licence page for details.

## References

1. UBC Master of Data Science Program. *DSCI 531: Effective Use of Visual Channels* – Lecture 2: Bar chart syntax. 2025.  

2. UBC Master of Data Science Program. *DSCI 531: Visualization for Communication* – Lecture 5: Axis label formatting. 2025.  

3. W3Schools. “CSS Color Names.” W3Schools.com. https://www.w3schools.com/cssref/css_colors.php (accessed 21 November 2025).  

4. City of Vancouver Open Data Portal. “Food Vendors” dataset. https://opendata.vancouver.ca/explore/dataset/food-vendors/table/ (accessed 21 November 2025).

