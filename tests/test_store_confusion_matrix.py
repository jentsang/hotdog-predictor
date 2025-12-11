"""
Add here the description
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import pytest

from src.analysis_tools import NaiveBayesModel


## ---------------------------------------------------------------------------
## You can either:
##   - Reuse the same DataFrames as in the store_raw_cv_scores tests
##     (test_normal_df, test_edge_case_df, test_exception_case_df),
##   - Or define similarly simple DataFrames here.
##
## ---------------------------------------------------------------------------

## ---------------------------------------------------------------------------
## Tests for store_confusion_matrix
## ---------------------------------------------------------------------------

def test_store_confusion_matrix_train_normal(tmp_path) -> None:
    """
    Normal case (train=True)

    Goal:
    - Call store_confusion_matrix(train=True) on a NaiveBayesModel that has
      a valid pipeline (no explicit fit required).
    - Verify that "train_confusion_matrix.png" is created and non-empty.

    Suggested steps:
    1. Create a small but normal training dataset with BUSINESS_NAME and
       is_hotdog labels (enough rows for 5-fold CV).
    2. Create a simple test set as well.
    3. Instantiate NaiveBayesModel with tmp_path as the base for output dirs.
    4. Ensure the figure_output_path directory exists.
    5. Call model.store_confusion_matrix(train=True).
    6. Build the expected path to "train_confusion_matrix.png".
    7. Assert that the file exists and that its size is greater than 0.
    """
    pass


def test_store_confusion_matrix_test_edge(tmp_path) -> None:
    """
    Edge case (train=False)

    Goal:
    - Call store_confusion_matrix(train=False) on a NaiveBayesModel that
      has been fitted on a small dataset.
    - Verify that "test_confusion_matrix.png" is created and non-empty.

    Suggested steps:
    1. Create a small training and test dataset.
    2. Instantiate NaiveBayesModel with tmp_path as output base.
    3. Call model.fit() so that is_fitted is True.
    4. Ensure the figure_output_path directory exists.
    5. Call model.store_confusion_matrix(train=False).
    6. Build the expected path to "test_confusion_matrix.png".
    7. Assert that the file exists and that its size is greater than 0.
    """
    pass


def test_store_confusion_matrix_exception(tmp_path) -> None:
    """
    Exception case

    Goal:
    - Call store_confusion_matrix(train=False) on a NaiveBayesModel that
      has NOT been fitted.
    - Verify that an AssertionError is raised due to check_if_fitted().

    Suggested steps:
    1. Create a small training and test dataset.
    2. Instantiate NaiveBayesModel with tmp_path as output base.
    3. Do NOT call model.fit().
    4. Use pytest.raises(AssertionError) as a context manager around
       model.store_confusion_matrix(train=False) to assert that an
       AssertionError is thrown.
    """
    pass


