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
## Tests for store_model_mismatches
## ---------------------------------------------------------------------------

def test_store_model_mismatches_train_normal(tmp_path) -> None:
    """
    Normal case (train=True)

    Goal:
    - Call store_model_mismatches(train=True) on a NaiveBayesModel with a
      normal training set.
    - Verify that "train__model_mismatches.csv" is created and (likely)
      contains at least one row of mismatches.

    Suggested steps:
    1. Create a small training and test dataset where some mistakes are
       likely (e.g., mixed BUSINESS_NAME texts).
    2. Instantiate NaiveBayesModel with tmp_path as output base.
       (No explicit fit() is required for train=True; it uses CV.)
    3. Ensure the table_output_path directory exists.
    4. Call model.store_model_mismatches(train=True).
    5. Build the expected path to "train__model_mismatches.csv".
    6. Assert that the file exists and that its size is greater than 0.
       (Optionally, check that the number of rows is <= training set size.)
    """
    pass


def test_store_model_mismatches_test_edge(tmp_path) -> None:
    """
    Edge case (train=False)

    Goal:
    - Call store_model_mismatches(train=False) on a NaiveBayesModel that has
      been fitted and whose test set is very easy (possibly no mismatches).
    - Verify that "test__model_mismatches.csv" is created, and possibly that
      it is empty except for the header.

    Suggested steps:
    1. Create a training and test dataset that is easy for the model
       (e.g., very clear hotdog vs not-hotdog names).
    2. Instantiate NaiveBayesModel with tmp_path as output base.
    3. Call model.fit() to train the model.
    4. Ensure the table_output_path directory exists.
    5. Call model.store_model_mismatches(train=False).
    6. Build the expected path to "test__model_mismatches.csv".
    7. Assert that the file exists.
       Optionally, read it with pandas and assert that the number of rows
       is 0 (no mismatches) or very small.
    """
    pass


def test_store_model_mismatches_exception(tmp_path) -> None:
    """
    Exception case

    Goal:
    - Call store_model_mismatches(train=False) on a NaiveBayesModel that has
      NOT been fitted.
    - Verify that an AssertionError is raised because predict() calls
      check_if_fitted() and is_fitted is still False.

    Suggested steps:
    1. Create a small training and test dataset.
    2. Instantiate NaiveBayesModel with tmp_path as output base.
    3. Do NOT call model.fit().
    4. Ensure the table_output_path directory exists.
    5. Use pytest.raises(AssertionError) as a context manager around
       model.store_model_mismatches(train=False) to assert that an
       AssertionError is thrown.
    """
    pass
