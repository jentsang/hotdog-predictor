"""
Command-line entry point for training and analyzing a Decision Tree
bag-of-words classifier.

This script loads training and testing data, fits an
`analysis_tools.TreeModel`, and exports cross-validation scores,
confusion matrices, mismatches, and the actual decision tree diagram. 
It is intended to be run from the command line.
"""

import click
from analysis_tools import TreeModel ## Import the Decision Tree model class
from pandas import read_csv, DataFrame, Series


@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--testing-data', type=str, help="Path to testing data")
@click.option('--figures-to', type=str, help="Path to where figures will be saved")
@click.option('--tables-to', type=str, help="Path to where tables will be saved")
@click.option('--seed', type=int, help="Random seed", default=522)
def main(training_data, testing_data, figures_to, tables_to, seed) -> None:
    """
    Run the end-to-end decision tree text classification pipeline.

    The command reads training and testing CSV files, extracts the
    ``BUSINESS_NAME`` column as text features and ``is_hotdog`` as the
    target, initializes an :class:`analysis_tools.TreeModel`, and
    generates evaluation artifacts such as cross-validation scores and
    confusion matrices.

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
    seed : int
        Random seed used when initializing the underlying model.

    Returns
    -------
    None

    Examples
    --------
    From the command line (assuming you are in ./scripts/):

    .. code-block:: bash

       python script_name.py \\
           --training-data=../data/train.csv \\
           --testing-data=../data/test.csv \\
           --figures-to=../results/figures/ \\
           --tables-to=../results/tables/ \\
           --seed=522
    """

    ## Here we are extracting the data from the source and filling na's with empty strings

    train_data = read_csv(training_data)
    test_data = read_csv(testing_data)

    X_train = train_data["BUSINESS_NAME"].fillna('')
    y_train = train_data["is_hotdog"]

    X_test = test_data["BUSINESS_NAME"].fillna('')
    y_test = test_data["is_hotdog"]

    ## Here we are declaring the model

    dt = TreeModel(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
        table_output_directory=tables_to,
        figure_output_directory=figures_to
    )

    ## Here we are extracting the CV scores

    dt.store_raw_cv_scores()

    dt.store_agg_cv_scores()

    ## Here we are extracting the model mismatches

    dt.store_confusion_matrix()
    
    dt.store_model_mismatches()

    # fitting the model and getting the decision tree diagram 
    dt.fit()

    dt.store_decision_tree_diagram()

    click.echo(f"Decision Tree tables and figures have been generated")

    return None



if __name__ == "__main__":
    main()