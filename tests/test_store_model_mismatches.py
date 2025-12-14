"""
Unit tests for NaiveBayesModel.store_model_mismatches.

This module defines three tests for the store_model_mismatches method:
1. Normal case (train=True):
   - Calls store_model_mismatches(train=True) on a NaiveBayesModel with valid
     training and test data.
   - Verifies that a non-empty "train__model_mismatches.csv" file is written.
2. Edge case (train=False):
   - Fits the model on an "easy" dataset and then calls
     store_model_mismatches(train=False).
   - Verifies that a "test__model_mismatches.csv" file is written and that the
     number of rows is not larger than the test set size.
3. Exception case (train=False without fitting):
   - Calls store_model_mismatches(train=False) without calling fit().
   - Verifies that an AssertionError is raised due to check_if_fitted().
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pytest

from src.analysis_tools import NaiveBayesModel


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

# "Easy" dataset: very clear hotdog vs not-hotdog names
easy_names = [
    "hotdog stand 1",
    "hotdog stand 2",
    "hotdog cart",
    "best hotdogs",
    "salad bar 1",
    "green salad shop",
    "veggie salad place",
    "healthy salad bowl",
] * 3  # repeat to have enough rows

easy_labels = [
    True,
    True,
    True,
    True,
    False,
    False,
    False,
    False,
] * 3

test_easy_df: pd.DataFrame = pd.DataFrame({
    "BUSINESS_NAME": easy_names,
    "is_hotdog": easy_labels,
})

# Tiny dataset for exception case (size doesn't matter much; we just don't fit)
test_tiny_df: pd.DataFrame = pd.DataFrame({
    "BUSINESS_NAME": ["dog house", "pizza slice"],
    "is_hotdog": [True, False],
})


## ---------------------------------------------------------------------------
## Tests for store_model_mismatches
## ---------------------------------------------------------------------------

def test_store_model_mismatches_train_normal(tmp_path) -> None:
    """
    Normal case (train=True)
    Call store_model_mismatches(train=True) on a NaiveBayesModel with a valid
    training and test set, and check that "train__model_mismatches.csv" is
    created and non-empty.
    """
    # 70/30 train-test split on the normal dataset
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

    # Ensure the table output directory exists
    os.makedirs(nb_model.table_output_path, exist_ok=True)

    # For train=True, the method should internally use CV on the training data
    nb_model.store_model_mismatches(train=True)

    train_mm_path: str = os.path.join(
        nb_model.table_output_path,
        "train__model_mismatches.csv",
    )

    assert os.path.isfile(train_mm_path)
    assert os.path.getsize(train_mm_path) > 0


def test_store_model_mismatches_test_edge(tmp_path) -> None:
    """
    Edge case (train=False)
    Fit the model on an "easy" dataset and call store_model_mismatches(train=False)
    to ensure that "test__model_mismatches.csv" is created and that it has
    at most as many rows as the test set.
    """
    # 70/30 train-test split on the easy dataset
    cut: int = len(test_easy_df) * 7 // 10

    X_train: pd.Series = test_easy_df["BUSINESS_NAME"].iloc[:cut]
    y_train: pd.Series = test_easy_df["is_hotdog"].iloc[:cut]

    X_test: pd.Series = test_easy_df["BUSINESS_NAME"].iloc[cut:]
    y_test: pd.Series = test_easy_df["is_hotdog"].iloc[cut:]

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

    # Create directories needed for saving model and tables
    os.makedirs(nb_model.model_output_path, exist_ok=True)
    os.makedirs(nb_model.table_output_path, exist_ok=True)

    # Fit the model so that is_fitted becomes True
    nb_model.fit()

    nb_model.store_model_mismatches(train=False)

    test_mm_path: str = os.path.join(
        nb_model.table_output_path,
        "test__model_mismatches.csv",
    )

    assert os.path.isfile(test_mm_path)

    # Optionally check that the number of mismatches is not larger than test set
    mismatches_df = pd.read_csv(test_mm_path)
    assert len(mismatches_df) <= len(X_test)


def test_store_model_mismatches_exception(tmp_path) -> None:
    """
    Exception case
    Call store_model_mismatches(train=False) on a NaiveBayesModel that has NOT
    been fitted and verify that an AssertionError is raised.
    """
    X_train: pd.Series = test_tiny_df["BUSINESS_NAME"]
    y_train: pd.Series = test_tiny_df["is_hotdog"]

    # Reuse the same tiny data for testing
    X_test: pd.Series = test_tiny_df["BUSINESS_NAME"]
    y_test: pd.Series = test_tiny_df["is_hotdog"]

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

    # Do NOT call nb_model.fit() here; we expect an AssertionError
    with pytest.raises(AssertionError):
        nb_model.store_model_mismatches(train=False)
