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
## Tests for store_agg_cv_scores
## ---------------------------------------------------------------------------

def test_store_agg_cv_scores_normal(tmp_path) -> None:
    """
    Normal case

    Goal:
    - Create a NaiveBayesModel with a reasonably sized training set.
    - Call store_agg_cv_scores().
    - Check that "agg_cv_scores.csv" exists and is non-empty.

    Suggested steps (for you to implement):
    1. Create a small DataFrame with BUSINESS_NAME and is_hotdog columns.
    2. Do a simple train/test split (e.g., 70/30) to get X_train, y_train,
       X_test, y_test.
    3. Instantiate NaiveBayesModel, using tmp_path as the base directory for
       table_output_directory, figure_output_directory, and model_output_directory.
    4. Ensure the table_output_path directory exists (os.makedirs).
    5. Call model.store_agg_cv_scores().
    6. Build the expected path to "agg_cv_scores.csv" inside table_output_path.
    7. Assert that the file exists and that its size is greater than 0.
    """
    pass


def test_store_agg_cv_scores_edge(tmp_path) -> None:
    """
    Edge case

    Goal:
    - Use training data that contains many empty BUSINESS_NAME values or
      very short text, but is still large enough for 5-fold CV.
    - Call store_agg_cv_scores().
    - Verify that the method runs and still produces a non-empty CSV.

    Suggested steps:
    1. Create a DataFrame where BUSINESS_NAME alternates between "" and some
       short token (e.g., "dog"), with enough rows for 5-fold CV (>= 10 or so).
    2. Split into X_train, y_train, X_test, y_test.
    3. Instantiate NaiveBayesModel with tmp_path as output base.
    4. Ensure the table_output_path directory exists.
    5. Call model.store_agg_cv_scores().
    6. Build the expected path to "agg_cv_scores.csv".
    7. Assert that the file exists and that its size is greater than 0.
    """
    pass


def test_store_agg_cv_scores_exception(tmp_path) -> None:
    """
    Exception case

    Goal:
    - Use a training set that is too small for 5-fold cross-validation
      (e.g., fewer than 5 samples), or completely empty.
    - Call store_agg_cv_scores().
    - Verify that a ValueError is raised because CV cannot run.

    Suggested steps:
    1. Create a very small DataFrame for training (e.g., 0–3 rows) with
       BUSINESS_NAME and is_hotdog columns.
    2. Create some simple test data (can reuse normal data or keep it tiny).
    3. Instantiate NaiveBayesModel with this tiny training set and tmp_path
       as output base.
    4. Ensure the table_output_path directory exists.
    5. Use pytest.raises(ValueError) as a context manager around
       model.store_agg_cv_scores() to assert that a ValueError is thrown.
    """
    pass
