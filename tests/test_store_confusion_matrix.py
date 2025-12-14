"""
Unit tests for NaiveBayesModel.store_confusion_matrix.

This module defines three tests for the store_confusion_matrix method:
1. Normal case (train=True):
   - Calls store_confusion_matrix(train=True) on a NaiveBayesModel with valid
     training and test data.
   - Verifies that a non-empty "train_confusion_matrix.png" file is written.
2. Edge case (train=False):
   - Fits the model and then calls store_confusion_matrix(train=False).
   - Verifies that a non-empty "test_confusion_matrix.png" file is written.
3. Exception case (train=False without fitting):
   - Calls store_confusion_matrix(train=False) without calling fit().
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
## Tests for store_confusion_matrix
## ---------------------------------------------------------------------------

def test_store_confusion_matrix_train_normal(tmp_path) -> None:
    """
    Normal case (train=True)
    Call store_confusion_matrix(train=True) on a NaiveBayesModel with a valid
    training and test set, and check that "train_confusion_matrix.png" is
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

    # Ensure the figure output directory exists
    os.makedirs(nb_model.figure_output_path, exist_ok=True)

    # For train=True, the method should internally use CV on the training data
    nb_model.store_confusion_matrix(train=True)

    train_cm_path: str = os.path.join(
        nb_model.figure_output_path,
        "train_confusion_matrix.png",
    )

    assert os.path.isfile(train_cm_path)
    assert os.path.getsize(train_cm_path) > 0


def test_store_confusion_matrix_test_edge(tmp_path) -> None:
    """
    Edge case (train=False)
    Fit the model on a small dataset and call store_confusion_matrix(train=False)
    to ensure that "test_confusion_matrix.png" is created and non-empty.
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

    # Create directories needed for saving model and figures
    os.makedirs(nb_model.model_output_path, exist_ok=True)
    os.makedirs(nb_model.figure_output_path, exist_ok=True)

    # Fit the model so that is_fitted becomes True
    nb_model.fit()

    nb_model.store_confusion_matrix(train=False)

    test_cm_path: str = os.path.join(
        nb_model.figure_output_path,
        "test_confusion_matrix.png",
    )

    assert os.path.isfile(test_cm_path)
    assert os.path.getsize(test_cm_path) > 0


def test_store_confusion_matrix_exception(tmp_path) -> None:
    """
    Exception case
    Call store_confusion_matrix(train=False) on a NaiveBayesModel that has NOT
    been fitted and verify that an AssertionError is raised.
    """
    # Use a tiny dataset; the size does not matter much here
    X_train: pd.Series = test_normal_df["BUSINESS_NAME"]
    y_train: pd.Series = test_normal_df["is_hotdog"]

    # Reuse part of the data for testing
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

    os.makedirs(nb_model.figure_output_path, exist_ok=True)

    # Do NOT call nb_model.fit() here; we expect an AssertionError
    with pytest.raises(AssertionError):
        nb_model.store_confusion_matrix(train=False)
