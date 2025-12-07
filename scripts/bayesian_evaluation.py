"""
Command-line entry point for evaluating the best Bayesian 
bag-of-words classifier.

This script loads training and testing data, fits an
`analysis_tools.NaiveBayesModel`, and exports cross-validation scores,
confusion matrices, and coefficient tables. It is intended to be run from the command line.
"""

import click
from analysis_tools import NaiveBayesModel ## Import the Naive Bayes Model class
from pandas import read_csv, DataFrame, Series
from scipy.stats import loguniform, randint


@click.command()
@click.option('--training-data', type=str, help="Path to training data", required=True)
@click.option('--testing-data', type=str, help="Path to testing data", required=True)
@click.option('--figures-to', type=str, help="Path to where figures will be saved", required=True)
@click.option('--tables-to', type=str, help="Path to where tables will be saved", required=True)
@click.option('--model-to', type=str, help="Path to where model will be saved", required=True)
@click.option('--seed', type=int, help="Random seed", default=522)
def main(training_data, testing_data, figures_to, tables_to, model_to, seed) -> None:
    """
    Run the end-to-end Naive Bayes text classification pipeline.

    The command reads training and testing CSV files, extracts the
    ``BUSINESS_NAME`` column as text features and ``is_hotdog`` as the
    target, initializes an :class:`analysis_tools.NaiveBayesModel`, and
    generates evaluation artifacts such as randomized search results,
    confusion matrices, and tables of misclassified examples.

    Parameters
    ----------
    training_data : str
        Path to the CSV file containing the training data. The file must
        include at least the columns ``BUSINESS_NAME`` and ``is_hotdog``.
    testing_data : str
        Path to the CSV file containing the test data. The file must
        include at least the columns ``BUSINESS_NAME`` and ``is_hotdog``.
    figures_to : str
        Directory where generated figures (e.g., confusion matrices) will
        be saved.
    tables_to : str
        Directory where generated tables (e.g., randomized search results
        and misclassified examples) will be saved.
    model_to : str
        Directory where the fitted bayesian model and related
        serialized objects will be saved.
    seed : int
        Random seed used when initializing the underlying model.

    Returns
    -------
    None

    Examples
    --------
    From the command line (assuming you are in ./scripts/):

    .. code-block:: bash

       python bayesian_evaluation.py \\
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
    ...     seed=522,
    ... )
    """

    # Extract data from the source and fill NA's with empty strings
    train_data = read_csv(training_data)
    test_data = read_csv(testing_data)

    X_train = train_data["BUSINESS_NAME"].fillna('')
    y_train = train_data["is_hotdog"]

    X_test = test_data["BUSINESS_NAME"].fillna('')
    y_test = test_data["is_hotdog"]

    # Declare the model
    nb = NaiveBayesModel(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        random_state=seed,
        table_output_directory=tables_to,
        figure_output_directory=figures_to,
        model_output_directory=model_to
    )

    param_grid = {
        "countvectorizer__max_features": randint(5, 95),
        "bernoullinb__alpha": loguniform(1e-2, 1e2),
    }
    
    # Perform and extract randomized search results
    nb.fit_randomized_CV(param_grid)
    nb.store_rcv_scores()

    # Extract confusion matrix and model mismatches
    nb.store_best_rcv_confusion_matrix()
    nb.store_best_rcv_model_mismatches()

    click.echo("Naive Bayes evaluation documents have been updated")

    return None


if __name__ == "__main__":
    main()