#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`decision_stump`
==================

.. module:: decision_stump
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2014-08-31, 01:52

"""


from warnings import warn
from operator import itemgetter
import concurrent.futures as cfut

import psutil
import numpy as np
from scipy.sparse import issparse

from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state, check_array
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree

try:
    import skboost.stumps.ext.classifiers as c_classifiers
except ImportError as e:
    c_classifiers = None

_all__ = [
    "NMMDecisionStump",
]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE


class DecisionStump(DecisionTreeClassifier):
    """A decision tree classifier.

    Parameters
    ----------
    criterion : string, optional (default="gini")
        Not used in Stratos Decision Stump.

    max_features : int, float, string or None, optional (default=None)
        Not used in Stratos Decision Stump.

    max_depth : integer or None, optional (default=None)
        Not used in Stratos Decision Stump. Always a depth 1 tree.

    min_samples_split : integer, optional (default=2)
        Not used in Stratos Decision Stump.

    min_samples_leaf : integer, optional (default=1)
        Not used in Stratos Decision Stump.

    random_state : int, RandomState instance or None, optional (default=None)
        Not used in Stratos Decision Stump. Nothing random in learning.

    Attributes
    ----------
    `tree_` : Tree object
        The underlying Tree object.

    `classes_` : array of shape = [n_classes] or a list of such arrays
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    `n_classes_` : int or list
        Alwats 2 fr this class.

    """

    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
        min_density=None,
        compute_importances=None,
        distributed_learning=True,
        calculate_probabilites=False,
        method="bp",
    ):
        super(DecisionStump, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state,
        )
        if min_density is not None:
            warn(
                "The min_density parameter is deprecated as of version 0.14 " "and will be removed in 0.16.",
                DeprecationWarning,
            )

        if compute_importances is not None:
            warn(
                "Setting compute_importances is no longer required as "
                "version 0.14. Variable importances are now computed on the "
                "fly when accessing the feature_importances_ attribute. "
                "This parameter will be removed in 0.16.",
                DeprecationWarning,
            )

        self.distributed_learning = distributed_learning
        self.calculate_probabilites = calculate_probabilites
        self.method = method

    def fit(self, X, y, sample_mask=None, X_argsorted=None, check_input=True, sample_weight=None):

        # Deprecations
        if sample_mask is not None:
            warn(
                "The sample_mask parameter is deprecated as of version 0.14 " "and will be removed in 0.16.",
                DeprecationWarning,
            )

        # Convert data
        random_state = check_random_state(self.random_state)
        if check_input:
            X = check_array(X, dtype=DTYPE, accept_sparse="csc")
            if issparse(X):
                X.sort_indices()
                if X.indices.dtype != np.intc or X.indptr.dtype != np.intc:
                    raise ValueError("No support for np.int64 index based " "sparse matrices")

        # Determine output settings
        n_samples, self.n_features_ = X.shape
        is_classification = isinstance(self, ClassifierMixin)

        y = np.atleast_1d(y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]

        if is_classification:
            y = np.copy(y)

            self.classes_ = []
            self.n_classes_ = []

            for k in range(self.n_outputs_):
                classes_k, y[:, k] = np.unique(y[:, k], return_inverse=True)
                self.classes_.append(classes_k)
                self.n_classes_.append(classes_k.shape[0])

        else:
            self.classes_ = [None] * self.n_outputs_
            self.n_classes_ = [1] * self.n_outputs_

        self.n_classes_ = np.array(self.n_classes_, dtype=np.intp)
        max_depth = 1
        max_features = 10

        if getattr(y, "dtype", None) != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if len(y) != n_samples:
            raise ValueError("Number of labels=%d does not match " "number of samples=%d" % (len(y), n_samples))
        if self.min_samples_split <= 0:
            raise ValueError("min_samples_split must be greater than zero.")
        if self.min_samples_leaf <= 0:
            raise ValueError("min_samples_leaf must be greater than zero.")
        if max_depth <= 0:
            raise ValueError("max_depth must be greater than zero. ")

        if sample_weight is not None:
            if getattr(sample_weight, "dtype", None) != DOUBLE or not sample_weight.flags.contiguous:
                sample_weight = np.ascontiguousarray(sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more " "than one dimension: %d" % len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError(
                    "Number of weights=%d does not match " "number of samples=%d" % (len(sample_weight), n_samples)
                )

        if self.method == "bp":
            self.tree_ = _fit_binary_decision_stump_breakpoint(
                X, y, sample_weight, X_argsorted, self.calculate_probabilites
            )
        elif self.method == "bp_threaded":
            self.tree_ = _fit_binary_decision_stump_breakpoint_threaded(
                X, y, sample_weight, X_argsorted, self.calculate_probabilites
            )
        else:
            self.tree_ = _fit_binary_decision_stump_breakpoint(
                X, y, sample_weight, X_argsorted, self.calculate_probabilites
            )

        if self.n_outputs_ == 1:
            self.n_classes_ = self.n_classes_[0]
            self.classes_ = self.classes_[0]

        return self

    def predict(self, X, check_input=True):
        """Predict class or regression value for X.

        For a classification model, the predicted class for each sample in X is
        returned. For a regression model, the predicted value based on X is
        returned.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes, or the predict values.
        """
        if getattr(X, "dtype", None) != DTYPE or X.ndim != 2:
            X = check_array(X, dtype=DTYPE)

        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise Exception("Tree not initialized. Perform a fit first")

        if self.n_features_ != n_features:
            raise ValueError(
                "Number of features of the model must "
                " match the input. Model n_features is %s and "
                " input n_features is %s " % (self.n_features_, n_features)
            )

        if self.tree_.get("direction") > 0:
            return ((X[:, self.tree_.get("best_dim")] > self.tree_.get("threshold")) * 2) - 1
        else:
            return ((X[:, self.tree_.get("best_dim")] <= self.tree_.get("threshold")) * 2) - 1

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. Classes are ordered
            by arithmetical order.
        """
        if getattr(X, "dtype", None) != DTYPE or X.ndim != 2:
            X = check_array(X, dtype=DTYPE)

        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise Exception("Tree not initialized. Perform a fit first.")

        if self.n_features_ != n_features:
            raise ValueError(
                "Number of features of the model must "
                " match the input. Model n_features is %s and "
                " input n_features is %s " % (self.n_features_, n_features)
            )

        proba = np.array(self.tree_["probabilities"]).take(self.predict(X) > 0, axis=0)

        if self.n_outputs_ == 1:
            proba = proba[:, : self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer

            return proba

        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, : self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba


def _fit_binary_decision_stump_breakpoint(X, y, sample_weight, argsorted_X=None, calculate_probabilities=False):
    Y = (y.flatten() * 2) - 1

    results = {"min_value": None, "best_dim": 0, "threshold": 0, "direction": 0, "probabilities": []}

    if sample_weight is None:
        sample_weight = np.ones(shape=(X.shape[0],), dtype="float") / (X.shape[0],)
    else:
        sample_weight /= np.sum(sample_weight)

    classifier_result = []

    for dim in range(X.shape[1]):
        if argsorted_X is not None:
            sorted_x = X[argsorted_X[:, dim], dim]
            w = sample_weight[argsorted_X[:, dim]]
            sorted_y = Y[argsorted_X[:, dim]]
        else:
            data_order = np.argsort(X[:, dim])
            sorted_x = X[data_order, dim]
            w = sample_weight[data_order]
            sorted_y = Y[data_order]

        breakpoint_indices = np.where(np.diff(sorted_x))[0] + 1

        w_pos_c = (w * (sorted_y > 0)).cumsum()
        w_neg_c = (w * (sorted_y < 0)).cumsum()

        left_errors = w_pos_c[breakpoint_indices] - w_neg_c[breakpoint_indices] + w_neg_c[-1]
        right_errors = w_neg_c[breakpoint_indices] - w_pos_c[breakpoint_indices] + w_pos_c[-1]
        best_left_point = np.argmin(left_errors)
        best_right_point = np.argmin(right_errors)
        if best_left_point < best_right_point:
            output = [
                dim,
                left_errors[best_left_point],
                (sorted_x[breakpoint_indices[best_left_point] + 1] + sorted_x[breakpoint_indices[best_left_point]]) / 2,
                1,
            ]
        else:
            output = [
                dim,
                right_errors[best_right_point],
                (sorted_x[breakpoint_indices[best_right_point] + 1] + sorted_x[breakpoint_indices[best_right_point]])
                / 2,
                -1,
            ]

        classifier_result.append(output)
        del sorted_x, sorted_y, left_errors, right_errors, w, w_pos_c, w_neg_c

    # Sort the returned data after lowest error.
    classifier_result = sorted(classifier_result, key=itemgetter(1))
    best_result = classifier_result[0]

    results["best_dim"] = int(best_result[0])
    results["min_value"] = float(best_result[1])
    # If the data is in integers, then set the threshold in integer as well.
    if X.dtype.kind in ("u", "i"):
        results["threshold"] = int(best_result[2])
    else:
        results["threshold"] = float(best_result[2])
    # Direction is defined as 1 if the positives labels are at
    # higher values and -1 otherwise.
    results["direction"] = int(best_result[3])

    if calculate_probabilities:
        results["probabilities"] = _calculate_probabilities(X[:, results["best_dim"]], Y, results)

    return results


def _fit_binary_decision_stump_breakpoint_threaded(
    X, y, sample_weight, argsorted_X=None, calculate_probabilities=False
):
    Y = y.flatten() * 2 - 1

    results = {"min_value": None, "best_dim": 0, "threshold": 0, "direction": 0, "probabilities": []}

    if sample_weight is None:
        sample_weight = np.ones(shape=(X.shape[0],), dtype="float") / (X.shape[0],)
    else:
        sample_weight /= np.sum(sample_weight)

    classifier_result = []

    tpe = cfut.ThreadPoolExecutor(max_workers=psutil.cpu_count())
    futures = []
    if argsorted_X is not None:
        for dim in range(X.shape[1]):
            futures.append(
                tpe.submit(_breakpoint_learn_one_dimension, dim, X[:, dim], Y, sample_weight, argsorted_X[:, dim])
            )
    else:
        for dim in range(X.shape[1]):
            futures.append(tpe.submit(_breakpoint_learn_one_dimension, dim, X[:, dim], Y, sample_weight))
    for future in cfut.as_completed(futures):
        classifier_result.append(future.result())

    # Sort the returned data after lowest error.
    classifier_result = sorted(classifier_result, key=itemgetter(1))
    best_result = classifier_result[0]

    results["best_dim"] = int(best_result[0])
    results["min_value"] = float(best_result[1])
    # If the data is in integers, then set the threshold in integer as well.
    if X.dtype.kind in ("u", "i"):
        results["threshold"] = int(best_result[2])
    else:
        results["threshold"] = float(best_result[2])
    # Direction is defined as 1 if the positives labels are at
    # higher values and -1 otherwise.
    results["direction"] = int(best_result[3])

    if calculate_probabilities:
        results["probabilities"] = _calculate_probabilities(X[:, results["best_dim"]], Y, results)

    return results


def _calculate_probabilities(X, Y, results):
    if results["direction"] > 0:
        labels = X > results["threshold"]
    else:
        labels = X <= results["threshold"]

    n_correct_negs = sum(Y[-labels] < 0)
    n_false_negs = sum(Y[-labels] > 0)
    n_false_pos = sum(Y[labels] < 0)
    n_correct_pos = sum(Y[labels] > 0)
    return [[n_correct_negs / len(Y), n_false_negs / len(Y)], [n_false_pos / len(Y), n_correct_pos / len(Y)]]


def _breakpoint_learn_one_dimension(dim_nbr, x, y, sample_weights, sorting_argument=None):
    if sorting_argument is None:
        sorting_argument = np.argsort(x)
    sorted_x = x[sorting_argument]
    w = sample_weights[sorting_argument]
    sorted_y = y[sorting_argument]

    breakpoint_indices = np.where(np.diff(sorted_x))[0] + 1

    w_pos_c = (w * (sorted_y > 0)).cumsum()
    w_neg_c = (w * (sorted_y < 0)).cumsum()

    left_errors = w_pos_c[breakpoint_indices] - w_neg_c[breakpoint_indices] + w_neg_c[-1]
    right_errors = w_neg_c[breakpoint_indices] - w_pos_c[breakpoint_indices] + w_pos_c[-1]
    best_left_point = np.argmin(left_errors)
    best_right_point = np.argmin(right_errors)

    if best_left_point < best_right_point:
        output = [
            dim_nbr,
            left_errors[best_left_point],
            (sorted_x[breakpoint_indices[best_left_point] - 1] + sorted_x[breakpoint_indices[best_left_point]]) / 2,
            1,
        ]
    else:
        output = [
            dim_nbr,
            right_errors[best_right_point],
            (sorted_x[breakpoint_indices[best_right_point] + 1] + sorted_x[breakpoint_indices[best_right_point]]) / 2,
            -1,
        ]

    return output
