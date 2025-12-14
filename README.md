# Food Vendors in Vancouver: Hot Dog or Not?

Authors: Zaki Aslam, Hector Palafox Prieto, Jennifer Tsang, and Samrawit Mezgebo Tsegay

## About

In this project, we used decision trees, logistic regression, and a Naive Bayes classifier to identify whether or not a food vendor sells hot dogs via their name. We trained each model individually using a cross-validation setup, and we compared the scores of the accuracy in order to determine a model to train and to compare to the test data. In our current results, a tuned Naïve Bayes classifier performs best, with test accuracy around 0.79 and relatively few false positives. Finally, we validated it with our test data and came to the conclusion that even though it is good enough for classifying most of the cases, it still struggles to discern from the minority class, which, in our case, is our target. 

The data set used in this project is Street food vending created by the City of Vancouver (Vancouver 2025b). It was sourced from the City of Vancouver Open Data Portal (Vancouver 2025a) and can be found [here](https://opendata.vancouver.ca/explore/dataset/food-vendors/table/). Each row in the dataset represents a food vendor and includes information such as the business name, description, and location. In this project, we derive the binary target from the description column (“Hot Dogs” vs other descriptions) and use the business name as the main predictor in our models.

## Report

The final report is available in the `reports/` folder:

- HTML: [dog_or_not_report.html](https://github.com/jentsang/hotdog-predictor/blob/main/reports/dog_or_not_report.html)  
- PDF:  [dog_or_not_report.pdf](https://github.com/jentsang/hotdog-predictor/blob/main/reports/dog_or_not_report.pdf)

## Dependencies

- [Docker](https://www.docker.com/)
- [VS Code](https://code.visualstudio.com/download)
- [VS Code Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

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


## Run locally with conda-lock

From the **repo root**, create and activate the environment from the lock file:

```bash
conda-lock install --name dsci522proj conda-lock.yml
conda activate dsci522proj
```

Then run the full workflow using the Makefile:

```bash
make clean
make all
```

> If you’re on Windows and `make` isn’t available, you’ll need to install it (or run the scripts manually—see the section below).


## Manual script execution (optional)

If you want to run individual steps manually, the scripts live in `scripts/` and can be executed one-by-one (e.g., `download_data.py`, `prepare_data.py`, model scripts, etc.).  
However, for grading/reproducibility, we recommend using the Makefile targets above.

---

## Running tests

After setting up the environment (either Docker or conda), run:

```bash
pytest tests/
```

To run one test file:

```bash
pytest tests/test_store_raw_cv_scores.py
```

## License

The written report and documentation contained in this repository are licensed under the Creative Commons Attribution–NonCommercial–NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license. See the LICENSE file for more information. If re-using or re-mixing the report, please provide attribution and a link to this repository.

The software code contained within this repository is licensed under the MIT License. See the LICENSE file for more information.

The Food Vendors dataset is provided by the City of Vancouver under the Open Government Licence – Vancouver. See the City of Vancouver open data licence page for details.


## References

1. UBC Master of Data Science Program. *DSCI 531: Effective Use of Visual Channels* – Lecture 2: Bar chart syntax. 2025.  

2. UBC Master of Data Science Program. *DSCI 531: Visualization for Communication* – Lecture 5: Axis label formatting. 2025.  

3. City of Vancouver Open Data Portal. “Food Vendors” dataset. https://opendata.vancouver.ca/explore/dataset/food-vendors/table/ (accessed 21 November 2025).

