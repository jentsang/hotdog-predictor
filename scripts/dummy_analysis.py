"""
Command-line entry point for training and analyzing a Dummy
bag-of-words classifier.

This script loads training and testing data, fits an
`analysis_tools.DummyModel()`, and exports the corresponding cross-validation scores.
It is intended to be run from the command line.
"""

import click
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pandas import read_csv, DataFrame
from src.analysis_tools import DummyModel 

@click.command()
@click.option('--training-data', type=str, help="Path to training data", required=True)
@click.option('--testing-data', type=str, help="Path to testing data", required=True)
@click.option('--tables-to', type=str, help="Path to where tables will be saved", required=True)
@click.option('--seed', type=int, help="Random seed", default=522)
def main(training_data, testing_data, tables_to, seed) -> None:
    """
    Run the end-to-end dummy text classification pipeline.
    
    The command reads training and testing CSV files, extracts the
    ``BUSINESS_NAME`` column as text features and ``is_hotdog`` as the
    target, initializes a :class:`analysis_tools.DummyModel`, and
    exports cross-validation scores for this simple baseline model.
    As this is just a dummy classifier, no figures or additional
    evaluation artifacts are generated.

    Parameters
    ----------
    training_data : str
        Path to the CSV file containing the training data. The file must
        include at least the columns ``BUSINESS_NAME`` and ``is_hotdog``.
    testing_data : str
        Path to the CSV file containing the test data. The file must
        include at least the columns ``BUSINESS_NAME`` and ``is_hotdog``.
    tables_to : str
        Directory where generated tables (e.g., cross-validation scores)
        will be saved.
    seed : int
        Random seed used when initializing the underlying model.

    Returns
    -------
    None

    Examples
    --------
    From the command line (assuming you are in ./scripts/):

    .. code-block:: bash

       python dummy_analysis.py \\
           --training-data=../data/processed/vendors_train.csv \\
           --testing-data=../data/processed/vendors_test.csv \\
           --tables-to=../results/tables/ \\
           --seed=522
    """

    # Extract data from the source and fill NA's with empty strings
    train_data = read_csv(training_data)
    test_data = read_csv(testing_data)

    X_train = train_data["BUSINESS_NAME"].fillna('')
    y_train = train_data["is_hotdog"]

    X_test = test_data["BUSINESS_NAME"].fillna('')
    y_test = test_data["is_hotdog"]

    # Declare the dummy baseline model
    dummy = DummyModel(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
        table_output_directory=tables_to,
    )

    # Export raw and aggregated cross-validation scores
    dummy.store_raw_cv_scores()
    dummy.store_agg_cv_scores()

    click.echo("Dummy cross-validation tables have been generated")

    return None


if __name__ == "__main__":
    main()