"""
Unit tests for NaiveBayesModel.store_raw_cv_scores.

This module defines three tests for the store_raw_cv_scores method:

1. Normal case:
   - A reasonably sized training set with varied BUSINESS_NAME strings.
   - Verifies that a non-empty "raw_cv_scores.csv" file is written.

2. Edge case:
   - A small training set with many empty BUSINESS_NAME values.
   - Verifies that the method still runs and writes a non-empty CSV.

3. Exception case:
   - An empty training set.
   - Verifies that calling store_raw_cv_scores raises a ValueError due
     to cross-validation being impossible with zero samples.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pytest

from src.analysis_tools import NaiveBayesModel


## ---------------------------------------------------------------------------
## Test data creation
## ---------------------------------------------------------------------------

# Normal-case dataset: mixed BUSINESS_NAME strings and is_hotdog labels
test_normal_df: pd.DataFrame = pd.DataFrame({
    "BUSINESS_NAME": [
        "dog dog dog dog",
        "tamales",
        "",
        "yellow chicken",
        "golden duck",
        "burger",
        "mexican taco",
        "dog galaxy",
        "the dog palace",
        "Thai chicken",
        "korean taco",
        "",
        "dogs",
        "Hot Dogs",
        "",
        "pizza burger",
        "hawt dawgs",
        "pizza dogs",
        "planet pizza",
        "what is up dog",
    ],
    "is_hotdog": [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        True,
    ],
})

# Edge-case dataset: alternating empty and non-empty BUSINESS_NAME values
test_edge_case_df: pd.DataFrame = pd.DataFrame({
    "BUSINESS_NAME": ["", "dog"] * 15,
    "is_hotdog": [True, False] * 15,
})

# Exception-case dataset: completely empty training data
test_exception_case_df: pd.DataFrame = pd.DataFrame({
    "BUSINESS_NAME": [],
    "is_hotdog": [],
})


## ---------------------------------------------------------------------------
## Tests for store_raw_cv_scores
## ---------------------------------------------------------------------------

def test_store_raw_cv_scores_normal(tmp_path) -> None:
    """
    Normal case

    Given a NaiveBayesModel with a built pipeline (via its __init__) and a
    small training set of BUSINESS_NAME strings with binary is_hotdog labels,
    calling store_raw_cv_scores() should run 5-fold CV and write a non-empty
    "NaiveBayes/raw_cv_scores.csv" file to the tables directory.
    """
    # 70/30 train-test split
    cut: int = len(test_normal_df) * 7 // 10

    X_train: pd.Series = test_normal_df["BUSINESS_NAME"].iloc[:cut]
    y_train: pd.Series = test_normal_df["is_hotdog"].iloc[:cut]

    X_test: pd.Series = test_normal_df["BUSINESS_NAME"].iloc[cut:]
    y_test: pd.Series = test_normal_df["is_hotdog"].iloc[cut:]

    base_dir: str = str(tmp_path) + os.sep

    nb_model: NaiveBayesModel = NaiveBayesModel(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        table_output_directory=base_dir,
        figure_output_directory=base_dir,
        model_output_directory=base_dir,
        random_state=522,
    )

    # Ensure the output directory exists
    os.makedirs(nb_model.table_output_path, exist_ok=True)

    # Call the method under test
    nb_model.store_raw_cv_scores()

    # Check that the expected file exists and is non-empty
    cv_scores_path: str = os.path.join(
        nb_model.table_output_path,
        "raw_cv_scores.csv",
    )

    assert os.path.isfile(cv_scores_path)
    assert os.path.getsize(cv_scores_path) > 0


def test_store_raw_cv_scores_edge(tmp_path) -> None:
    """
    Edge case

    Given a NaiveBayesModel with many empty BUSINESS_NAME values in the
    training set, calling store_raw_cv_scores() should still successfully run
    5-fold CV and write a non-empty "NaiveBayes/raw_cv_scores.csv" file.
    """
    # 70/30 train-test split on the edge-case dataset
    cut: int = len(test_edge_case_df) * 7 // 10

    X_train: pd.Series = test_edge_case_df["BUSINESS_NAME"].iloc[:cut]
    y_train: pd.Series = test_edge_case_df["is_hotdog"].iloc[:cut]

    X_test: pd.Series = test_edge_case_df["BUSINESS_NAME"].iloc[cut:]
    y_test: pd.Series = test_edge_case_df["is_hotdog"].iloc[cut:]

    base_dir: str = str(tmp_path) + os.sep

    nb_model: NaiveBayesModel = NaiveBayesModel(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        table_output_directory=base_dir,
        figure_output_directory=base_dir,
        model_output_directory=base_dir,
        random_state=522,
    )

    os.makedirs(nb_model.table_output_path, exist_ok=True)

    nb_model.store_raw_cv_scores()

    cv_scores_path: str = os.path.join(
        nb_model.table_output_path,
        "raw_cv_scores.csv",
    )

    assert os.path.isfile(cv_scores_path)
    assert os.path.getsize(cv_scores_path) > 0


def test_store_raw_cv_scores_exception(tmp_path) -> None:
    """
    Exception case

    Given a NaiveBayesModel with an empty training set, calling
    store_raw_cv_scores() should raise a ValueError because 5-fold
    cross-validation cannot be performed with zero samples.
    """
    X_train: pd.Series = test_exception_case_df["BUSINESS_NAME"]
    y_train: pd.Series = test_exception_case_df["is_hotdog"]

    # Test data can be anything; cross-validation only uses training splits.
    # Here we reuse the normal dataset for simplicity.
    X_test: pd.Series = test_normal_df["BUSINESS_NAME"]
    y_test: pd.Series = test_normal_df["is_hotdog"]

    base_dir: str = str(tmp_path) + os.sep

    nb_model: NaiveBayesModel = NaiveBayesModel(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        table_output_directory=base_dir,
        figure_output_directory=base_dir,
        model_output_directory=base_dir,
        random_state=522,
    )

    os.makedirs(nb_model.table_output_path, exist_ok=True)

    # With an empty training set, cross_validate inside store_raw_cv_scores()
    # should raise a ValueError (e.g., n_splits > n_samples).
    with pytest.raises(ValueError):
        nb_model.store_raw_cv_scores()
