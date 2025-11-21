# Food Vendors in Vancouver: Hot Dog or Not?

author: TSANG JENNIFER, TSEGAY SAMRAWIT MEZGEBO, ASLAM ZAKI, PALAFOX PRIETO HECTOR  

Demo of a data analysis project for DSCI 522 (Data Science Workflows), a course in the Master of Data Science program at the University of British Columbia.

## About

In this project, we use the City of Vancouver “Food Vendors” open dataset to explore where different kinds of food vendors operate in Vancouver and whether we can automatically identify hot-dog vendors.

Our main research question is:

> Can we predict whether a food vendor is a hot-dog vendor or not?

For Milestone 1, we are primarily using the **business name** of each vendor to decide whether they are a hot-dog vendor or not, and we encode this as a binary target variable (“hot-dog vendor” vs “not hot-dog vendor”). As the project develops, we may also incorporate other attributes as features if they improve the model and remain easy to interpret.

The purpose of this project is to:

- practice building and evaluating a simple binary classifier on real open data,  
- work with text fields (business names) to construct a meaningful target variable,  
- follow a clear and reproducible workflow that other users can re-run.

Once the analysis is complete, we will update this section with a brief summary of the final model performance and main findings.

## Data

The data set used in this project is the **“Food Vendors”** dataset from the City of Vancouver Open Data Portal:

- Dataset: Food Vendors  
- Provider: City of Vancouver Open Data Portal  
- URL: https://opendata.vancouver.ca/explore/dataset/food-vendors/table/  

Each row in the dataset represents a food vendor and includes information such as the business name, description, and location. In this project, we use the business name to derive a binary label indicating whether the vendor appears to be a hot-dog vendor, and we plan to explore how this relates to other available variables.

The data is published under the **Open Government Licence – Vancouver**.  
For details, see: https://opendata.vancouver.ca/pages/licence/

## Repository structure

For Milestone 1, the repository includes the following:

- `README.md` – project overview and instructions  
- `CODE_OF_CONDUCT.md` – project code of conduct  
- `CONTRIBUTING.md` – guidelines for contributing to the project  
- `data/` – folder for the Food Vendors dataset and any derived data files  
- `notebooks/dog_or_not.ipynb` – main analysis notebook  
- `environment.yml` – base conda environment specification  
- `conda-*.lock` – conda lock files for major operating systems (e.g. `conda-linux-64.lock`, `conda-osx-64.lock`, `conda-win-64.lock`)  
- `LICENSE` – licenses for the code, report, and data  

## Report

For Milestone 1, the main analysis is contained in the notebook:

- `notebooks/dog_or_not.ipynb`

In later milestones, we may render this analysis to an HTML or PDF report and link it here.

## Usage

From the root of this repository, the analysis can be run as follows (assuming `conda` is installed).

First time setup (recommended, using an OS-specific conda lock file):

    git clone https://github.com/jentsang/DSCI-522-Group-17.git
    cd DSCI-522-Group-17

    # Choose the lock file that matches your operating system and run ONE of:
    conda-lock install --name dsci522proj conda-linux-64.lock   # Linux
    conda-lock install --name dsci522proj conda-osx-64.lock     # macOS (Intel)
    conda-lock install --name dsci522proj conda-win-64.lock     # Windows

    conda activate dsci522proj

Alternatively, you can create the environment directly from `environment.yml`:

    conda env create -f environment.yml
    conda activate dsci522proj

To run the analysis:

    jupyter lab

Then, in JupyterLab, open `notebooks/dog_or_not.ipynb` and run the cells in order to reproduce the analysis.

If the project is later converted to a Quarto document, the analysis can be rendered with:

    quarto render path/to/analysis.qmd

## Dependencies

The main dependencies needed to run the analysis are:

- Python 3.11  
- conda  
- conda-lock  
- jupyterlab  
- scikit-learn  
- pandas  
- plus any additional packages listed in `environment.yml`

The `environment.yml` file defines the base environment, and the OS-specific conda lock files (`conda-linux-64.lock`, `conda-osx-64.lock`, `conda-win-64.lock`) pin exact package versions for reproducibility across different systems.

## License

The written report and documentation contained in this repository are licensed under the **Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0)** license. See the `LICENSE` file for more information. If re-using or re-mixing the report, please provide attribution and a link to this repository.

The software code contained within this repository is licensed under the **MIT License**. See the `LICENSE` file for more information.

The Food Vendors dataset is provided by the City of Vancouver under the **Open Government Licence – Vancouver**. See the City of Vancouver open data licence page for details.
