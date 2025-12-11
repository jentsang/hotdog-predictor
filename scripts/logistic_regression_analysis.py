"""
Command-line entry point for training and analyzing a logistic regression
bag-of-words classifier.

This script loads training and testing data, fits an
`analysis_tools.LRModel`, and exports cross-validation scores,
confusion matrices, coefficient tables, and plots of discriminative
features. It is intended to be run from the command line.
"""

import click
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.analysis_tools import LRModel ## Import the logistic regression class
from pandas import read_csv, DataFrame, Series


@click.command()
@click.option('--training-data', type=str, help="Path to training data", required=True)
@click.option('--testing-data', type=str, help="Path to testing data", required=True)
@click.option('--figures-to', type=str, help="Path to where figures will be saved", required=True)
@click.option('--tables-to', type=str, help="Path to where tables will be saved", required=True)
@click.option('--model-to', type=str, help="Path to where the model will be saved", required=True)
@click.option('--seed', type=int, help="Random seed", default=522)
def main(training_data, testing_data, figures_to, tables_to, model_to, seed) -> None:
    """
    Run the end-to-end logistic regression text classification pipeline.

    The command reads training and testing CSV files, extracts the
    ``BUSINESS_NAME`` column as text features and ``is_hotdog`` as the
    target, initializes an :class:`analysis_tools.LRModel`, and
    generates evaluation artifacts such as cross-validation scores,
    confusion matrices, coefficient summaries, and a serialized model
    file.

    Parameters
    ----------
    training_data : str
        Path to the CSV file containing the training data. The file must
        include at least the columns ``BUSINESS_NAME`` and ``is_hotdog``.
    testing_data : str
        Path to the CSV file containing the test data. The file must
        include at least the columns ``BUSINESS_NAME`` and ``is_hotdog``.
    figures_to : str
        Directory where generated figures (e.g., confusion matrices and
        discriminant feature plots) will be saved.
    tables_to : str
        Directory where generated tables (e.g., cross-validation scores
        and coefficient tables) will be saved.
    model_to : str
        Directory where the fitted logistic regression model and related
        serialized objects will be saved.
    seed : int
        Random seed used when initializing the underlying model.

    Returns
    -------
    None

    Examples
    --------
    From the command line (assuming you are in ./scripts/):

    $ python logistic_regression_analysis.py \\
        --training-data=../data/processed/vendors_train.csv \\
        --testing-data=../data/processed/vendors_test.csv \\
        --figures-to=../results/figures/ \\
        --tables-to=../results/tables/ \\
        --model-to=../results/models/ \\
        --seed=522

    When called programmatically:

    >>> main(
    ...     training_data="data/train.csv",
    ...     testing_data="data/test.csv",
    ...     figures_to="results/figures/",
    ...     tables_to="results/tables/",
    ...     model_to="results/models/",
    ...     seed=522,
    ... )
    """

    ## Here we are extracting the data from the source and filling na's with empty strings

    train_data = read_csv(training_data)
    test_data = read_csv(testing_data)

    X_train = train_data["BUSINESS_NAME"].fillna('')
    y_train = train_data["is_hotdog"]

    X_test = test_data["BUSINESS_NAME"].fillna('')
    y_test = test_data["is_hotdog"]

    ## Here we are declaring the model

    lr = LRModel(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
        table_output_directory=tables_to,
        figure_output_directory=figures_to,
        model_output_directory=model_to
    )

    ## Here we are extracting the CV scores

    lr.store_raw_cv_scores()

    lr.store_agg_cv_scores()

    ## Here we are extracting the model mismatches

    lr.store_confusion_matrix()
    
    lr.store_model_mismatches()

    ## Here we are fitting and extracting the coefficients, and their most relevant features
    ## Note that this only applies to the Logistic Regression, given the nature of 
    ## having interpretable coefficients.

    lr.fit()

    lr.store_lr_coefficient_table()

    lr.store_most_discriminant_features()

    click.echo("Logistic Regression documents have been updated")

    return None



if __name__ == "__main__":
    main()