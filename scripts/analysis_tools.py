"""
Utility classes and helpers for the hot-dog predictor text classification.

This module implements a small framework around scikit-learn models to
make more efficient the cross-validation, evaluation, and export of results
for the bag-of-words (BOW) text classifiers. It provides a hierarchy of model
classes that wrap common estimators such as dummy baselines, decision
trees, logistic regression, and naive Bayes classifiers.
"""


import numpy as np
import pandas as pd
import altair as alt
import pickle
import json
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split, cross_validate,
    cross_val_predict, RandomizedSearchCV
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from scipy.stats import loguniform, randint
from sklearn.metrics import ConfusionMatrixDisplay
from mglearn.tools import visualize_coefficients






class BOWModel():
    """
    Base class for bag-of-words text classification models.

    This class stores the training and test splits and provides utilities
    to construct vocabulary and contains the models defined for the analysis.
    It is designed to be subclassed by concrete models that define a particular
    scikit-learn estimator and pipeline.

    Parameters
    ----------
    X_train : pandas.Series
        Training data.
    y_train : pandas.Series
        Training outputs.
    X_test : pandas.Series
        Test data.
    y_test : pandas.Series
        Test outputs.


    Attributes
    ----------
    models : dict of str to type of sklearn.base.BaseEstimator
        Registry mapping model names to scikit-learn estimator classes.
    pipeline : sklearn.pipeline.Pipeline or None
        Classification pipeline.
    vocab : list of str or None
        Vocabulary learned from the training data via `CountVectorizer`.
    model_name : str or None
        Name of the estimator to use, as a key in `models`.
    model : type of sklearn.base.BaseEstimator or None
        Estimator class selected from `models`.
    X_train : pandas.Series
        Stored training data.
    y_train : pandas.Series
        Stored training outputs.
    X_test : pandas.Series
        Stored test data.
    y_test : pandas.Series
        Stored test outputs.

    Examples
    --------
    Create a base model and extract a vocabulary:

    >>> bow = BOWModel(X_train, y_train, X_test, y_test)
    >>> vocab = bow.fetch_vocab()
    >>> "dog" in vocab
    True
    """
    

    models: dict[str, type[BaseEstimator]] = {
        "Dummy": DummyClassifier,
        "DecisionTree": DecisionTreeClassifier,
        "LogisticRegression": LogisticRegression,
        "NaiveBayes": BernoulliNB
    }

    def __init__(
            self,
            X_train: pd.Series,
            y_train: pd.Series,
            X_test: pd.Series,
            y_test: pd.Series,
        ) -> None:
        """
        Initialize the base bag-of-words model.

        Stores the train/test splits and initializes common attributes
        shared by all subclasses.

        Parameters
        ----------
        X_train : pandas.Series
            Stored training data.
        y_train : pandas.Series
            Stored training outputs.
        X_test : pandas.Series
            Stored test data.
        y_test : pandas.Series
            Stored test outputs.

        Returns
        -------
        None
        """

        self.pipeline: Pipeline | None = None
        self.vocab: list[str] | None = None
        self.model_name: str | None = None
        self.model: BaseEstimator | None = None
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        return None

    def assign_model(self) -> type[BaseEstimator] | None:
        """
        Assign the estimator class corresponding to ``self.model_name``.

        Returns
        -------
        type of sklearn.base.BaseEstimator
            The estimator class associated with ``self.model_name``.

        Raises
        ------
        ValueError
            If ``self.model_name`` is not a key in ``BOWModel.models``.
        """

        if not self.model_name in BOWModel.models:
            raise ValueError("The model name provided is not defined for this analysis")
        
        return BOWModel.models[self.model_name]
    
    
    def check_pipeline(self) -> None:
        """
        Check that the pipeline has been defined.

        Raises
        ------
        AssertionError
            If ``self.pipeline`` is ``None``.
        """
        
        assert not self.pipeline is None, "Pipeline has not been defined yet"
        
        return None
    
    def fetch_vocab(
            self,
            save_vocab: bool = False,
            path: str = "../results/tables/vocab.csv"
        ) -> list[str]:
        """
        Learn a bag-of-words vocabulary from the training data.

        A binary ``CountVectorizer`` is fitted on ``X_train`` and the
        resulting vocabulary is stored in ``self.vocab``. Optionally, the
        vocabulary is also stored as a CSV file.

        Parameters
        ----------
        save_vocab : bool, default=False
            If True, save the vocabulary to a CSV file.
        path : str, default='../results/tables/vocab.csv'
            Output path for the vocabulary CSV when ``save_vocab`` is True.

        Returns
        -------
        list of str
            List of tokens in the learned vocabulary.

        Examples
        --------
        >>> bow = BOWModel(X_train, y_train, X_test, y_test)
        >>> vocab = bow.fetch_vocab(save_vocab=True,
        ...                         path="../results/tables/vocab.csv")
        >>> len(vocab) > 0
        True
        """
    
        vocab_pipeline: Pipeline = make_pipeline(
            CountVectorizer(binary=True)
        )
        
        vocab_pipeline.fit(self.X_train, self.y_train)
        
        self.vocab = list(
            vocab_pipeline.named_steps["countvectorizer"].get_feature_names_out()
        )

        if save_vocab:
            pd.DataFrame({"words": self.vocab}).to_csv(path, index=False)

        return self.vocab


class ActualModel(BOWModel):
    """
    Concrete bag-of-words model wrapping a scikit-learn estimator.

    This class extends :class:`BOWModel` with a pipeline definition
    (tokenization and vectorization followed by a classifier), along with
    utilities for cross-validation, prediction, and exporting evaluation
    artifacts such as scores, mismatches, and confusion matrices.

    Parameters
    ----------
    model_name : str
        Name of the estimator to use. Must be a key in
        `BOWModel.models`.
    table_output_directory : str, default="../results/tables/"
        Directory where tabular outputs (CSV files) will be written.
    figure_output_directory : str, default="../results/figures/"
        Directory where plots and figures will be written.
    model_output_directory : str, default="../results/models/"
        Directory where serialized model objects will be written.
    random_state : int, default=522
        Random seed used for randomized hyperparameter search and any
        underlying stochastic estimators.
    **kwargs
        Additional keyword arguments forwarded to :class:`BOWModel`,
        including ``X_train``, ``y_train``, ``X_test``, and ``y_test``.

    Attributes
    ----------
    random_state : int
        Stored random seed used by the model and randomized search.
    model_name : str
        Name of the estimator.
    model : type of sklearn.base.BaseEstimator
        Estimator class used in the pipeline.
    is_fitted : bool
        Flag indicating whether the pipeline has been fitted.
    table_output_path : str
        Base path for CSV outputs for this model.
    figure_output_path : str
        Base path for figure outputs for this model.
    model_output_path : str
        Base path for serialized model files for this model.
    scores : dict
        Dictionary for storing derived evaluation metrics.
    cv_raw_results : dict or None
        Raw cross-validation results returned by
        `sklearn.model_selection.cross_validate`.

    Examples
    --------
    >>> model = ActualModel(
    ...     model_name="LogisticRegression",
    ...     X_train=X_train,
    ...     y_train=y_train,
    ...     X_test=X_test,
    ...     y_test=y_test,
    ...     random_state=522,
    ... )
    >>> model.build_pipeline()
    >>> model.perform_cv()
    >>> model.fit()
    >>> y_pred = model.predict(X_test)
    """

    def __init__(
            self, model_name: str,
            table_output_directory: str = "../results/tables/",
            figure_output_directory: str = "../results/figures/",
            model_output_directory: str = "../results/models/",
            random_state: int = 522,
            **kwargs
        ) -> None:
        """
        Initialize an ``ActualModel`` instance.

        Parameters
        ----------
        model_name : str
            Name of the estimator to use, must be in ``BOWModel.models``.
        table_output_directory : str, default="../results/tables/"
            Directory where CSV outputs will be saved.
        figure_output_directory : str, default="../results/figures/"
            Directory where figures will be saved.
        model_output_directory : str, default="../results/models/"
            Directory where fitted models and best estimators will be
            serialized to disk.
        random_state : int, default=522
            Random seed used for randomized hyperparameter search and
            any stochastic components in the estimator.
        **kwargs
            Additional keyword arguments passed to :class:`BOWModel`
            (e.g., ``X_train``, ``y_train``, ``X_test``, ``y_test``).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``model_name`` is not a valid key in ``BOWModel.models``.
        """

        super().__init__(**kwargs)

        self.model_name = model_name
        self.model = self.assign_model()
        self.is_fitted: bool = False

        self.random_state: int = random_state

        self.random_search: RandomizedSearchCV | None = None
        self.is_rcv_fitted: bool = False

        self.table_output_path: str = table_output_directory + model_name
        self.figure_output_path: str = figure_output_directory + model_name
        self.model_output_path: str = model_output_directory + model_name

        self.scores: dict = {}
        self.cv_raw_results = None
        
        return None
    
    def build_pipeline(self) -> None:
        """
        Build the text classification pipeline.

        The pipeline consists of a binary :class:`CountVectorizer`
        followed by the estimator selected by ``self.model``.

        Returns
        -------
        None
        """

        self.pipeline: Pipeline = make_pipeline(
            CountVectorizer(binary=True),
            self.model()
        )

        return None
    
    def check_if_fitted(self) -> None:
        """
        Check that the pipeline has been defined and fitted.

        First validates that the pipeline exists, then checks the
        ``is_fitted`` flag.

        Raises
        ------
        AssertionError
            If the pipeline has not been built or ``is_fitted`` is False.
        """

        self.check_pipeline()

        assert self.is_fitted, "The model has not been fitted yet"

        return None
    
    def check_if_rcv_fitted(self) -> None:
        """
        Check that the randomized search has been run and fitted.

        This helper verifies that `fit_randomized_CV` has created a
        `sklearn.model_selection.RandomizedSearchCV` instance and
        that it has been successfully fitted.

        Raises
        ------
        AssertionError
            If the randomized search object does not exist or has not
            yet been fitted.
        """

        assert isinstance(self.random_search, RandomizedSearchCV), "The random search has not been performed yet"

        assert self.is_rcv_fitted, "The random search has not been fitted yet"

        return None
    
    
    
    def perform_cv(self) -> None:
        """
        Run cross-validation on the training data.

        Performs 5-fold cross-validation using the current pipeline and
        stores the raw results in ``self.cv_raw_results``.

        Returns
        -------
        None
        """

        self.check_pipeline()
        
        self.cv_raw_results = cross_validate(
            self.pipeline,
            self.X_train,
            self.y_train,
            cv=5,
            return_train_score=True
        )

        return None

    def cv_predict(self, prob: bool = False, as_list: bool = False) -> np.ndarray:
        """
        Generate cross-validated predictions on the training set.

        Uses :func:`sklearn.model_selection.cross_val_predict` with the
        current pipeline.

        Parameters
        ----------
        prob : bool, default=False
            If True, return class probabilities using ``predict_proba``.
            Otherwise, return class labels using ``predict``.
        as_list : bool, default=False
            If True, cast the predictions to a Python list before
            returning.

        Returns
        -------
        numpy.ndarray or list
            Array (or list) of predictions or class probabilities,
            depending on ``prob`` and ``as_list``.

        Raises
        ------
        AssertionError
            If the pipeline has not been built.
        """
        
        self.check_pipeline()

        method = "predict"

        if prob:
            method = "predict_proba"

        predictions: np.ndarray = cross_val_predict(
            self.pipeline,
            self.X_train,
            self.y_train,
            cv=5,
            method=method            
        )

        if as_list:
            return predictions.tolist()
        
        return predictions

    def fit(self) -> None:
        """
        Fit the pipeline on the training data.

        Sets the ``is_fitted`` flag to True after successful fitting and
        serializes the fitted pipeline to
        ``self.model_output_path + f'/{self.model_name}.pickle'``.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the pipeline has not been built.
        """
        
        self.check_pipeline()

        self.pipeline.fit(self.X_train, self.y_train)

        self.is_fitted = True

        with open(self.model_output_path + f'/{self.model_name}.pickle', 'wb') as f:
            pickle.dump(self.pipeline, f)

        return None
    
    def predict(self, X_data: pd.DataFrame | pd.Series, prob: bool = False) -> np.ndarray:
        """
        Make predictions or return probabilities for new data.

        Parameters
        ----------
        X_data : pandas.DataFrame or pandas.Series
            Input data to predict on.
        prob : bool, default=False
            If True, return class probabilities via ``predict_proba``.
            Otherwise, return predicted class labels.

        Returns
        -------
        numpy.ndarray
            Predicted labels or probabilities, depending on ``prob``.

        Raises
        ------
        AssertionError
            If the model has not been fitted.
        """

        self.check_if_fitted()

        if prob:
            return self.pipeline.predict_proba(X_data)

        return self.pipeline.predict(X_data)
    

    def fit_randomized_CV(
            self,
            param_grid: dict,
            iterations: int = 500,
            jobs: int = -1
        ) -> None:
        """
        Run a randomized hyperparameter search over the pipeline.

        This method wraps `sklearn.model_selection.RandomizedSearchCV`
        using the current text classification pipeline (vectorizer plus
        estimator), fits the search on the training data, and stores the
        best model and associated results on the instance. The best
        estimator is also serialized to
        ``self.model_output_path + f'/best_{self.model_name}.pickle'``.
        Additionally, the best score, best parameters, and the
        cross-validation results are serialized into a JSON file in the
        same directory.

        Parameters
        ----------
        param_grid : dict
            Mapping of hyperparameter names to distributions or lists of
            candidate values, as expected by `RandomizedSearchCV`.
        iterations : int, default=500
            Number of parameter settings that are sampled.
        jobs : int, default=-1
            Number of jobs to run in parallel. ``-1`` means using all
            available processors.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the pipeline has not been built yet.
        """

        self.check_pipeline()

        self.rcv_params: list = [f"param_{name}" for name in param_grid.keys()]

        self.random_search: RandomizedSearchCV = RandomizedSearchCV(
            self.pipeline,
            param_distributions=param_grid,
            n_jobs=jobs,
            n_iter=iterations,
            return_train_score=True,
            random_state=self.random_state
        )

        self.random_search.fit(self.X_train, self.y_train)

        self.is_rcv_fitted = True

        self.best_model: BaseEstimator = self.random_search.best_estimator_

        results_dict = {
            'best_score': self.random_search.best_score_,
            'best_params': self.random_search.best_params_
        }

        json_path = self.model_output_path + f'{self.model_name}_random_search_results.json'
        with open(json_path, 'w') as json_file:
            json.dump(results_dict, json_file, indent=4)

        with open(self.model_output_path + f'/best_{self.model_name}.pickle', 'wb') as f:
            pickle.dump(self.best_model, f)
        
        self.best_score: float = self.random_search.best_score_
        self.best_params: dict = self.random_search.best_params_
        self.rcv_results: pd.DataFrame = pd.DataFrame(self.random_search.cv_results_)



        return None
    
    
    def store_raw_cv_scores(self) -> None:
        """
        Save raw cross-validation scores.

        If cross-validation has not yet been run, it is performed first.
        The scores are written to
        ``self.table_output_path + "/raw_cv_scores.csv"``.

        Returns
        -------
        None
        """

        if self.cv_raw_results is None:
            self.perform_cv()

        pd.DataFrame(
            self.cv_raw_results
        ).round(4).to_csv(
            self.table_output_path + "/raw_cv_scores.csv",
            index=False
        )

        return None
    
    def store_agg_cv_scores(self) -> None:
        """
        Save aggregated cross-validation scores.

        Computes the mean and standard deviation of each score across
        folds and writes the result to
        ``self.table_output_path + "/agg_cv_scores.csv"``.

        Returns
        -------
        None
        """

        if self.cv_raw_results is None:
            self.perform_cv()

        pd.DataFrame(
            self.cv_raw_results
        ).agg(
            ['mean', 'std']
        ).round(3).T.to_csv(
            self.table_output_path + "/agg_cv_scores.csv",
        )

        return None
    
    def store_model_mismatches(self, train: bool = True) -> None:
        """
        Export misclassified examples for inspection.

        For training data, cross-validated predictions and probabilities
        are used. For test data, predictions from the fitted pipeline are
        used. Only rows where ``y != y_hat`` are saved.

        Parameters
        ----------
        train : bool, default=True
            If True, use training data and cross-validated predictions.
            If False, use test data and predictions from the fitted model.

        Returns
        -------
        None
        """

        self.check_pipeline()

        if train:

            fit_type: str = "train"

            data_dict: dict = {
                "y": self.y_train.to_list(),
                "y_hat": self.cv_predict(as_list=True),
                "probabilities": self.cv_predict(as_list=True, prob=True),
                "x": self.X_train.to_list(),
            }
        
        else:

            fit_type: str = "test"

            data_dict: dict = {
                "y": self.y_test.to_list(),
                "y_hat": list(self.predict(self.X_test)),
                "probabilities": list(self.predict(self.X_test, prob=True)),
                "x": self.X_test.to_list(),
            }
        
        df: pd.DataFrame = pd.DataFrame(data_dict)

        df[df["y"] != df["y_hat"]].sort_values('probabilities').to_csv(
            self.table_output_path + f'/{fit_type}__model_mismatches.csv',
            index=False
        )

        return None
    
    def store_confusion_matrix(self, train: bool = True) -> None:
        """
        Save a confusion matrix plot to disk.

        Uses :class:`sklearn.metrics.ConfusionMatrixDisplay` to compute
        and plot the confusion matrix for training or test data.

        Parameters
        ----------
        train : bool, default=True
            If True, compute the confusion matrix on training data using
            cross-validated predictions. If False, compute it on test
            data using predictions from the fitted model.

        Returns
        -------
        None
        """

        self.check_pipeline()

        disp: ConfusionMatrixDisplay | None = None

        if train:

            fit_type: str = "train"

            disp = ConfusionMatrixDisplay.from_predictions(
                self.y_train,
                self.cv_predict()
            )
        
        else:

            fit_type: str = "test"

            disp = ConfusionMatrixDisplay.from_predictions(
                self.y_test,
                self.predict(self.X_test)
            )
        
        disp.figure_.savefig(
            self.figure_output_path + f'/{fit_type}_confusion_matrix.png'
        )

        return None
    
    def store_rcv_scores(self, head: bool = True) -> None:
        """
        Save summary scores from the randomized search to CSV.

        The randomized search results stored in ``self.rcv_results`` are
        filtered to a subset of informative columns and written to disk.
        Optionally, only the top-ranked parameter settings are saved.

        The output is written to either
        ``self.table_output_path + "/RandomizedSearchCV_results_head.csv"``
        or
        ``self.table_output_path + "/RandomizedSearchCV_results_full.csv"``.

        Parameters
        ----------
        head : bool, default=True
            If True, save only the top 5 ranked parameter settings.
            If False, save the full results table.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the randomized search has not been run and fitted.
        """

        self.check_if_rcv_fitted()

        columns: list = [
            "rank_test_score",
            "mean_test_score",
            "mean_train_score",
            "mean_fit_time",
            "mean_score_time"
        ] + self.rcv_params

        results: pd.DataFrame = (
            self.rcv_results[columns]
            .set_index("rank_test_score")
            .sort_index()
        )

        if head:

            results.head(5).to_csv(
                self.table_output_path + '/RandomizedSearchCV_results_head.csv'
            )

        else:      

            results.to_csv(
                self.table_output_path + '/RandomizedSearchCV_results_full.csv'
            )

        return None
    
    def store_best_rcv_model_mismatches(self) -> None:
        """
        Export misclassified test examples for the best CV model.

        Uses the best estimator found by the randomized search to generate
        predictions and class probabilities on the test set. Only examples
        where ``y != y_hat`` are retained and written to

        ``self.table_output_path + "/best_rcv_model_test_mismatches.csv"``

        for later inspection.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the randomized search has not been run and fitted.
        """

        self.check_if_rcv_fitted()

        data_dict: dict = {
            "y": self.y_test.to_list(),
            "y_hat": list(self.best_model.predict(self.X_test)),
            "probabilities": list(self.best_model.predict_proba(self.X_test)[:, 1]),
            "x": self.X_test.to_list(),
        }
    # THE FIX: Select only the probability for the positive class (index 1)
        df: pd.DataFrame = pd.DataFrame(data_dict)

        df[df["y"] != df["y_hat"]].sort_values('probabilities').to_csv(
            self.table_output_path + "/best_rcv_model_test_mismatches.csv",
            index=False
        )

        return None
    
    def store_best_rcv_confusion_matrix(self) -> None:
        """
        Save a confusion matrix for the best CV model on the test set.

        Uses the best estimator obtained from the randomized search to
        compute a confusion matrix on ``X_test`` / ``y_test`` via
        `sklearn.metrics.ConfusionMatrixDisplay`. The resulting
        figure is written to

        ``self.figure_output_path + "/best_rcv_model_test_confusion_matrix.png"``

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the randomized search has not been run and fitted.
        """

        self.check_if_rcv_fitted()

        disp = ConfusionMatrixDisplay.from_predictions(
            self.y_test,
            self.best_model.predict(self.X_test)
        )
        
        disp.figure_.savefig(
            self.figure_output_path + "/best_rcv_model_test_confusion_matrix.png"
        )

        return None



class TreeModel(ActualModel):
    """
    Decision tree-based bag-of-words classifier.

    This subclass of `ActualModel` configures a
    `sklearn.tree.DecisionTreeClassifier` within the bag-of-words
    pipeline and provides helpers for inspecting the learned tree.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize a decision tree model.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :class:`ActualModel` and
            :class:`BOWModel`, typically including ``X_train``,
            ``y_train``, ``X_test``, ``y_test``, and optionally a
            ``random_state`` that is handled by :class:`ActualModel`.
        Returns
        -------
        None
        """

        super().__init__(model_name = "DecisionTree", **kwargs)

        self.build_pipeline()
        self.pipeline.named_steps["decisiontreeclassifier"].set_params(random_state=self.random_state)

        return None
    
    def tree_depth(self) -> int:
        """
        Return the maximum depth of the decision tree, and store it 
        in the model directory.

        Returns
        -------
        int
            Maximum depth of the underlying decision tree.

        Raises
        ------
        AssertionError
            If the model has not been fitted.
        """

        self.check_if_fitted()

        max_depth_dict = {
            "max_depth": self.pipeline["decisiontreeclassifier"].tree_.max_depth
        }

        json_path = self.model_output_path + f'{self.model_name}_max_depth.json'
        with open(json_path, 'w') as json_file:
            json.dump(max_depth_dict, json_file, indent=4)

        return self.pipeline["decisiontreeclassifier"].tree_.max_depth
    
    def store_decision_tree_diagram(self) -> None:
        """
        Save a diagram of the decision tree to disk.

        Uses `sklearn.tree.plot_tree` to visualize the top levels of
        the tree (up to depth 3) and writes the plot to
        ``self.figure_output_path + "/diagram.png"``.

        Returns
        -------
        None
        """

        self.check_if_fitted()

        plot_tree(
            self.pipeline.named_steps["decisiontreeclassifier"],
            feature_names=self.fetch_vocab(),
            max_depth=3,
            fontsize=7
        )

        plt.savefig(
            self.figure_output_path + "/diagram.png"
        )

        return None

class LRModel(ActualModel):
    """
    Logistic regression-based bag-of-words classifier.

    This subclass of `ActualModel` configures a
    `sklearn.linear_model.LogisticRegression` estimator and adds
    utilities for inspecting and exporting model coefficients.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize a logistic regression model.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to :class:`ActualModel` and
            :class:`BOWModel`, typically including ``X_train``,
            ``y_train``, ``X_test``, ``y_test``, and optionally a
            ``random_state`` that is handled by :class:`ActualModel`.

        Returns
        -------
        None
        """

        super().__init__(model_name = "LogisticRegression", **kwargs)

        self.build_pipeline()
        self.pipeline.named_steps["logisticregression"].set_params(random_state=self.random_state)

        return None

    
    def lr_intercept(self) -> float:
        """
        Return the intercept term of the logistic regression model, and
        stores it as a JSON in the model directory.

        Returns
        -------
        float
            Intercept of the logistic regression estimator.

        Raises
        ------
        AssertionError
            If the model has not been fitted.
        """

        self.check_if_fitted()

        lr_intercept_dict = {
            "intercept": self.pipeline.named_steps["logisticregression"].intercept_[0]
        }

        json_path = self.model_output_path + f'{self.model_name}_intercept.json'
        with open(json_path, 'w') as json_file:
            json.dump(lr_intercept_dict, json_file, indent=4)

        return self.pipeline.named_steps["logisticregression"].intercept_[0]
    
    def extract_coefficients(self) -> None:
        """
        Extract the logistic regression coefficients.

        The coefficients are stored in ``self.lr_coefficients`` as a
        one-dimensional NumPy array.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If the model has not been fitted.
        """

        self.check_if_fitted()

        self.lr_coefficients: np.ndarray = self.pipeline.named_steps["logisticregression"].coef_[0]

        return None

    
    def store_lr_coefficient_table(self) -> None:
        """
        Save a table of logistic regression coefficients.

        A CSV file mapping each token in the vocabulary to its
        corresponding coefficient is written to
        ``self.table_output_path + "/coefficients.csv"``.

        Returns
        -------
        None
        """

        self.extract_coefficients()

        df: pd.DataFrame = pd.DataFrame({
            "token": self.fetch_vocab(),
            "coefficients": self.lr_coefficients
        })

        df.to_csv(
            self.table_output_path + "/coefficients.csv",
            index=False
        )

        return None

    def store_most_discriminant_features(self) -> None:
        """
        Save a plot of the most discriminative features.

        Uses :func:`mglearn.tools.visualize_coefficients` to visualize
        the top 5 positive and negative coefficients and saves the figure
        to ``self.figure_output_path + "/most_discriminant_features.png"``.

        Returns
        -------
        None
        """

        self.extract_coefficients()

        visualize_coefficients(
            self.lr_coefficients,
            self.fetch_vocab(),
            n_top_features=5
        )
        
        plt.savefig(
            self.figure_output_path + "/most_discriminant_features.png"
        )

        return None
    
class NaiveBayesModel(ActualModel):
    """
    Naive Bayes-based bag-of-words classifier.

    This subclass of `ActualModel` configures a
    `sklearn.naive_bayes.BernoulliNB` estimator within the
    bag-of-words pipeline.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize a Naive Bayes model.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to `ActualModel` and
            `BOWModel`, including ``X_train``, ``y_train``,
            ``X_test``, and ``y_test``.

        Returns
        -------
        None
        """

        super().__init__(model_name = "NaiveBayes", **kwargs)

        self.build_pipeline()

        return None


class DummyModel(ActualModel):
    """
    Dummy baseline bag-of-words classifier.

    This subclass of `ActualModel` configures a
    `sklearn.dummy.DummyClassifier` as a simple baseline. Certain
    evaluation methods are intentionally disabled because they are not
    meaningful for a dummy model.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initialize a dummy baseline model.

        Parameters
        ----------
        **kwargs
            Keyword arguments forwarded to `ActualModel` and
            `BOWModel`, typically including ``X_train``,
            ``y_train``, ``X_test``, and ``y_test``.

        Returns
        -------
        None
        """

        super().__init__(model_name = "Dummy", **kwargs)

        self.build_pipeline()

        return None
    
    def cv_predict(self, *args, **kwargs) -> Exception:
        """
        Cross-validated predictions are not available for this model.

        Raises
        ------
        AttributeError
            Always raised to indicate that this method is disabled for
            ``DummyModel``.
        """
        raise AttributeError("This method is not available for this model")
    
    def store_model_mismatches(self, *args, **kwargs) -> Exception:
        """
        Exporting mismatches is not available for this model.

        Raises
        ------
        AttributeError
            Always raised to indicate that this method is disabled for
            ``DummyModel``.
        """
        raise AttributeError("This method is not available for this model")
    
    def store_confusion_matrix(self, *args, **kwargs) -> Exception:
        """
        Confusion matrix plots are not available for this model.

        Raises
        ------
        AttributeError
            Always raised to indicate that this method is disabled for
            ``DummyModel``.
        """
        raise AttributeError("This method is not available for this model")
    
if __name__== "__main__":
    pass







    

