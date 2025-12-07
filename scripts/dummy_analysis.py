"""
Command-line entry point for training and analyzing a Dummy
bag-of-words classifier.

This script loads training and testing data, fits an
`analysis_tools.DummyModel()`, and exports the corresponding cross-validation scores.
It is intended to be run from the command line.
"""

import click
from pandas import read_csv, DataFrame
from analysis_tools import DummyModel 

@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--testing-data', type=str, help="Path to testing data")
@click.option('--tables-to', type=str, help="Path to where tables will be saved")
def main(training_data, testing_data,tables_to) -> None:
    """
    Run the end-to-end Dummy text classification pipeline.
    
    The command reads training and testing CSV files, extracts the
    ``BUSINESS_NAME`` column as text features and ``is_hotdog`` as the
    target, initializes a :class:`analysis_tools.DummyModel`, and
    the cross validation scores. As this is just a dummy baseline model
    just the scores are exported with no other subsequent figures generated.

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

    Returns
    -------
    None

    Examples
    --------
    From the command line (assuming you are in ./scripts/):

    .. code-block:: bash

       python dummy_analysis.py \\
           --training-data=../data/processed/train.csv \\
           --testing-data=../data/processed/test.csv \\
           --tables-to=../results/tables/ 
    """

    ## Here we are extracting the data from the source and filling na's with empty strings

    train_data = read_csv(training_data)
    test_data = read_csv(testing_data)

    X_train = train_data["BUSINESS_NAME"].fillna('')
    y_train = train_data["is_hotdog"]

    X_test = test_data["BUSINESS_NAME"].fillna('')
    y_test = test_data["is_hotdog"]

    ## Here we are declaring the model

    dummy = DummyModel(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        table_output_directory=tables_to,
    )

    dummy.store_raw_cv_scores()
    dummy.store_agg_cv_scores()

    click.echo(f"Dummy tables have been generated")

if __name__ == "__main__":
    main()