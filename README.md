# DSCI 522 Group 17 – Food Vendors in Vancouver: Hot Dog or Not?

## Authors

- TSANG JENNIFER  
- TSEGAY SAMRAWIT MEZGEBO  
- ASLAM ZAKI  
- PALAFOX PRIETO HECTOR  

## Project overview

In this project, we use the City of Vancouver “Food Vendors” open dataset to study where different kinds of food vendors operate in Vancouver.

Our main question is:

> Can we predict whether a food vendor is a hot-dog vendor or not using its location and related attributes?

We create a binary target variable that labels each vendor as “hot-dog vendor” or “not hot-dog vendor” based on the description field. We then use information such as location and other available variables to build and evaluate a binary classification model.

The project focuses on:

- working with open data in a reproducible way  
- performing a basic binary classification task  
- documenting the workflow so others can re-run the analysis  

## Data

- Dataset: Food Vendors  
- Provider: City of Vancouver Open Data Portal  
- URL: https://opendata.vancouver.ca/explore/dataset/food-vendors/table/  

The dataset contains information about food vendors on Vancouver streets, including their locations and descriptions. We download the data and save it in the `data/` directory.

The data is published under the Open Government Licence – Vancouver.  
For details, see: https://opendata.vancouver.ca/pages/licence/

In this project, we also create a derived variable indicating whether a vendor appears to be a hot-dog vendor based on the description.

## Repository structure

For Milestone 1, the repository is organized as follows:

- `README.md` – project overview and instructions.  
- `LICENSE` – licenses for the code, report, and data.  
- `environment.yml` – conda environment file listing the main dependencies.  
- `notebooks/` – contains the main analysis notebook(s), e.g. `notebooks/dog_or_not.ipynb`.  
- `data/` – folder for the Food Vendors dataset and any derived data files.  

Additional folders and files (such as `src/`, `reports`, and a `Makefile`) may be added in later milestones.

## How to run the analysis

These steps assume that conda is installed.

1. Clone the repository and move into the project directory:

   git clone https://github.com/jentsang/DSCI-522-Group-17.git  
   cd DSCI-522-Group-17

2. Create the conda environment using the provided `environment.yml`:

   conda env create -f environment.yml

3. Activate the environment:

   conda activate dsci522proj

4. Start JupyterLab:

   jupyter lab

5. Run the analysis notebook:

   In JupyterLab, open `notebooks/dog_or_not.ipynb` and run the cells in order to reproduce the analysis.

If the project is later converted to a Quarto document, the analysis can be rendered with:

   quarto render path/to/analysis.qmd

## Dependencies

The main dependencies needed to run the analysis are:

- conda-lock  
- scikit-learn  
- pandas  
- jupyterlab  

These packages are specified in the `environment.yml` file under the `dsci522proj` conda environment. For the full and up-to-date list of packages and versions, please see `environment.yml` and any conda lock files associated with this project.

## Licenses

This project uses different licenses for the code, the report, and the data:

- Code: licensed under the MIT License.  
- Reports and written documentation: licensed under the Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license.  
- Data: the Food Vendors dataset is provided by the City of Vancouver under the Open Government Licence – Vancouver.

For full details, see the `LICENSE` file in this repository.
