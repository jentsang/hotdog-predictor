# Food Vendors in Vancouver: Hot Dog or Not?

author: TSANG JENNIFER, TSEGAY SAMRAWIT MEZGEBO, ASLAM ZAKI, PALAFOX PRIETO HECTOR  

Demo of a data analysis project for DSCI 522 (Data Science Workflows), a course in the Master of Data Science program at the University of British Columbia.

## About

In this project, we use the City of Vancouver “Food Vendors” open dataset to explore where different kinds of food vendors operate in Vancouver and whether we can automatically identify hot-dog vendors.

Our main research question is:

> Can we predict whether a food vendor is a hot-dog vendor or not?

For Milestone 1, we construct a binary target variable is_hotdog using the DESCRIPTION column in the data: it is True when DESCRIPTION is "Hot Dogs" and False otherwise. We then use the vendor’s BUSINESS_NAME as the main feature and train text-based classifiers to predict this label. As the project develops, we may also incorporate other attributes as features if they improve the model and remain easy to interpret.

The purpose of this project is to:

- practice building and evaluating a simple binary classifier on real open data,  
- work with text fields (business names) to construct a meaningful target variable,  
- follow a clear and reproducible workflow that other users can re-run.

In Milestone 1 we compare several models (Dummy, Decision Tree, Logistic Regression, and Naïve Bayes) and find that a tuned Naïve Bayes classifier performs best, achieving a test accuracy of about 0.79 with very few false positives.

In Milestone 2 we added data validation for the column and file names, as well as checking for the data types in our columns. Additionally, we created a Docker image for this project to be run in any device with the same environment and libraries as we have.

## Data

The data set used in this project is the **“Food Vendors”** dataset from the City of Vancouver Open Data Portal:

- Dataset: Food Vendors  
- Provider: City of Vancouver Open Data Portal  
- URL: https://opendata.vancouver.ca/explore/dataset/food-vendors/table/  

Each row in the dataset represents a food vendor and includes information such as the business name, description, and location. In this project, we derive the binary target is_hotdog from the DESCRIPTION column (“Hot Dogs” vs other descriptions) and use the BUSINESS_NAME as the main predictor in our models.

The data is published under the **Open Government Licence – Vancouver**.  
For details, see: https://opendata.vancouver.ca/pages/licence/

## Repository structure

For Milestone 1, the repository includes the following:

- `README.md` – project overview and instructions  
- `CODE_OF_CONDUCT.md` – project code of conduct  
- `CONTRIBUTING.md` – guidelines for contributing to the project  
- `data/` – folder for the Food Vendors dataset and any derived data files  
- `notebooks/dog_or_not.ipynb` – main analysis notebook (source)
- `notebooks/dog_or_not.html` – rendered HTML report
- `notebooks/dog_or_not.pdf` – rendered PDF report
- `environment.yml` – base conda environment specification (with pinned versions) 
- `conda-lock.yml` – conda lock file that pins exact package versions for reproducibility
- `LICENSE` – licenses for the code, report, and data  

## Report

For Milestone 1, the main analysis and report are contained in:
- source notebook: [`notebooks/dog_or_not.ipynb`](notebooks/dog_or_not.ipynb)
- rendered HTML report: [`notebooks/dog_or_not.html`](notebooks/dog_or_not.html)
- rendered PDF report: [`notebooks/dog_or_not.pdf`](notebooks/dog_or_not.pdf)

## Usage

### Working offline

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

If you want to run the docker image of this repo, you can instead just run:

```
docker pull jentsang/hotdog-predictor
```

And once the image is pulled, you can execute:

```
docker run --rm -it -p 8888:8888 \
  jentsang/hotdog-predictor \
  start-notebook.sh --NotebookApp.token='' --NotebookApp.password=''
```

Alternatively, you can also use the `docker-compose.yml` file available in the root by running:

```
docker-compose up
```

(Do not forget to close the container by using `Ctrl+C` and typing `docker-compose rm` into the terminal)

To run the analysis you can just type:

```http://localhost:8888/```

In your browser (make sure there are no other instances of jupyterlab open)

Or if you installed and executed the environment execute

```
jupyter lab
```

Then, in JupyterLab, open `notebooks/dog_or_not.ipynb` and run the cells in order to reproduce the analysis.

Next, under the "Kernel" menu click "Restart Kernel and Run All Cells...".

## Dependencies

The main dependencies needed to run the analysis (as specified in environment.yml) are:

- python 3.11
- conda-lock 3.0.4
- scikit-learn 1.7.2
- pandas 2.3.3
- jupyterlab 4.4.10
- altair 6.0.0
- pip 25.3
- pip:
    - mglearn 0.2.0

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

