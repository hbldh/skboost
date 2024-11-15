#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`regression_stump`
==================

.. module:: regression_stump
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import _tree

try:
    import skboost.stumps.ext.classifiers as c_classifiers
except ImportError as e:
    c_classifiers = None

__all__ = [
    "RegressionStump",
]

# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE


class RegressionStump(DecisionTreeRegressor):
    def __init__(
        self,
        criterion="mse",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
        min_density=None,
        compute_importances=None,
        method="default",
    ):
        super(RegressionStump, self).__init__(
            criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, random_state
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

        # Fixed point parameters.
        self.f_above = None
        self.f_below = None
        self.method = method

    def fit(self, X, y, sample_mask=None, X_argsorted=None, check_input=True, sample_weight=None):
        random_state = check_random_state(self.random_state)

        # Deprecations
        if sample_mask is not None:
            warn(
                "The sample_mask parameter is deprecated as of version 0.14 " "and will be removed in 0.16.",
                DeprecationWarning,
            )

        if X_argsorted is not None:
            warn(
                "The X_argsorted parameter is deprecated as of version 0.14 " "and will be removed in 0.16.",
                DeprecationWarning,
            )

        # Convert data
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
        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        if sample_weight is not None:
            if getattr(sample_weight, "dtype", None) != DOUBLE or not sample_weight.flags.contiguous:
                sample_weight = np.ascontiguousarray(sample_weight, dtype=DOUBLE)
            if len(sample_weight.shape) > 1:
                raise ValueError("Sample weights array has more " "than one dimension: %d" % len(sample_weight.shape))
            if len(sample_weight) != n_samples:
                raise ValueError(
                    "Number of weights=%d does not match " "number of samples=%d" % (len(sample_weight), n_samples)
                )

        if self.method == "default":
            self.tree_ = _fit_regressor_stump(X, y, sample_weight, X_argsorted)
        elif self.method == "threaded":
            self.tree_ = _fit_regressor_stump_threaded(X, y, sample_weight, X_argsorted)
        elif self.method == "c":
            self.tree_ = _fit_regressor_stump_c_ext(X, y, sample_weight, X_argsorted)
        elif self.method == "c_threaded":
            self.tree_ = _fit_regressor_stump_c_ext_threaded(X, y, sample_weight, X_argsorted)
        else:
            self.tree_ = _fit_regressor_stump(X, y, sample_weight, X_argsorted)

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
        X = check_array(X, dtype=DTYPE, accept_sparse="csr")
        if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
            raise ValueError("No support for np.int64 index based " "sparse matrices")

        n_samples, n_features = X.shape

        if self.tree_ is None:
            raise Exception("Tree not initialized. Perform a fit first")

        if self.n_features_ != n_features:
            raise ValueError(
                "Number of features of the model must "
                " match the input. Model n_features is %s and "
                " input n_features is %s " % (self.n_features_, n_features)
            )

        return self.tree_.get("coefficient") * (
            X[:, self.tree_.get("best_dim")] > self.tree_.get("threshold")
        ) + self.tree_.get("constant")

    def predict_labels(self, X):
        return ((self.predict(X) >= 0.0) * 2) - 1


def _fit_regressor_stump(X, y, sample_weight, argsorted_X=None):
    """Train a regression stump by weighted least-squares method.

    :param X: A [N x M] array, where N is the number of features and M is the number of samples.
    :type X: :py:class:`numpy.ndarray`
    :param y: A M long array or list of binary labels.
    :type y: :py:class:`numpy.ndarray`.
    :param sample_weight: The prior distribution for the examples. If None, a uniform prior
     distribution is used.
    :type sample_weight: :py:class:`numpy.ndarray`
    :param argsorted_X: Optional argument array that sorts ``X``.
    :type argsorted_X: :py:class:`numpy.ndarray`
    :returns: The required values for performing classification.
    :rtype: dict

    """
    Y = y.flatten()

    if sample_weight is None:
        sample_weight = np.ones(shape=(X.shape[0],), dtype="float") / (X.shape[0],)
    else:
        sample_weight /= np.sum(sample_weight)

    n_samples, n_dims = X.shape
    if X.dtype in ("float", "float32"):
        thresholds = np.zeros((n_dims,), dtype="float")
    else:
        thresholds = np.zeros((n_dims,), dtype="int")
    coeffs = np.zeros((n_dims,), dtype="float")
    constants = np.zeros((n_dims,), dtype="float")
    errors = np.zeros((n_dims,), dtype="float")

    # Iterate over all feature dimensions and train the optimal
    # regression stump for each dimension.
    for dim in range(n_dims):
        if argsorted_X is not None:
            data_order = argsorted_X[:, dim]
        else:
            data_order = np.argsort(X[:, dim])

        # Sort the weights and labels with argument for this dimension.
        # Time: 25%
        sorted_weights = sample_weight[data_order]
        sorted_output = Y[data_order]

        # Cumulative sum of desired output multiplied with weights.
        # Time: 10 %
        Szw = (sorted_weights * sorted_output).cumsum()
        # Cumulative sum of the weights.
        Sw = sorted_weights.cumsum()

        # Calculate regression function parameters.
        # Time: 25 %
        b = Szw / Sw
        zz = np.where((1.0 - Sw) < 1e-10)
        Sw[zz] = 0.0
        a = ((Szw[-1] - Szw) / (1 - Sw)) - b
        Sw[zz] = 1.0

        # Calculate the weighted square error:
        # Time: 40 %
        e = (
            (sorted_weights * (sorted_output * sorted_output)).sum()
            - (2 * a * (Szw[-1] - Szw))
            - (2 * b * Szw[-1])
            + ((a * a) + (2 * a * b)) * (1 - Sw)
            + (b * b)
        )

        del sorted_weights
        del sorted_output
        del Szw, Sw

        min_ind = e.argmin()
        errors[dim] = e[min_ind]
        del e
        coeffs[dim] = a[min_ind]
        del a
        constants[dim] = b[min_ind]
        del b

        # Handle floating point data different from integer data when it comes
        # to setting the threshold.
        if X.dtype in ("float", "float32"):
            if min_ind == (n_samples - 1):
                thresholds[dim] = X[data_order[min_ind], dim] + 0.1
            elif min_ind == 0:
                thresholds[dim] = X[data_order[min_ind], dim] - 0.1
            else:
                thresholds[dim] = (X[data_order[min_ind], dim] + X[data_order[min_ind + 1], dim]) / 2
        else:
            if min_ind == (n_samples - 1):
                thresholds[dim] = np.floor(X[data_order[min_ind], dim]) + 1
            elif min_ind == 0:
                thresholds[dim] = np.floor(X[data_order[min_ind], dim]) - 1
            else:
                v1 = int(X[data_order[min_ind], dim])
                v2 = int(X[data_order[min_ind + 1], dim])
                thr = (v1 + v2) / 2
                if np.abs(thr) > (2**31):
                    print("Threshold for dimension {0} was greater than 32 bit integer!".format(dim))
                thresholds[dim] = np.int32(thr)

        del data_order

    best_dim = errors.argmin()
    results = {
        "best_dim": int(best_dim),
        "min_value": float(errors[best_dim]),
        "threshold": float(thresholds[best_dim]),
        "coefficient": float(coeffs[best_dim]),
        "constant": float(constants[best_dim]),
    }

    return results


def _fit_regressor_stump_threaded(X, y, sample_weight, argsorted_X=None):
    Y = y.flatten()

    if sample_weight is None:
        sample_weight = np.ones(shape=(X.shape[0],), dtype="float") / (X.shape[0],)
    else:
        sample_weight /= np.sum(sample_weight)

    classifier_result = []

    with cfut.ThreadPoolExecutor(max_workers=psutil.cpu_count()) as tpe:
        futures = []
        if argsorted_X is not None:
            for dim in range(X.shape[1]):
                futures.append(
                    tpe.submit(_regressor_learn_one_dimension, dim, X[:, dim], Y, sample_weight, argsorted_X[:, dim])
                )
        else:
            for dim in range(X.shape[1]):
                futures.append(tpe.submit(_regressor_learn_one_dimension, dim, X[:, dim], Y, sample_weight))
        for future in cfut.as_completed(futures):
            classifier_result.append(future.result())

    # Sort the returned data after lowest error.
    classifier_result = sorted(classifier_result, key=itemgetter(1))
    best_result = classifier_result[0]

    return {
        "best_dim": int(best_result[0]),
        "min_value": float(best_result[1]),
        "threshold": float(best_result[2]),
        "coefficient": float(best_result[3]),
        "constant": float(best_result[4]),
    }


def _fit_regressor_stump_c_ext(X, y, sample_weight, argsorted_X=None):
    """Train a regression stump by weighted least-squares method.

    :param X: A [N x M] array, where N is the number of features and M is the number of samples.
    :type X: :py:class:`numpy.ndarray`
    :param y: A M long array or list of binary labels.
    :type y: :py:class:`numpy.ndarray`.
    :param sample_weight: The prior distribution for the examples. If None, a uniform prior
     distribution is used.
    :type sample_weight: :py:class:`numpy.ndarray`
    :param argsorted_X: Optional argument array that sorts ``X``.
    :type argsorted_X: :py:class:`numpy.ndarray`
    :returns: The required values for performing classification.
    :rtype: dict

    """
    if c_classifiers is None:
        return _fit_regressor_stump(X, y, sample_weight, argsorted_X)

    if sample_weight is None:
        sample_weight = np.ones(shape=(len(y),), dtype="float") / (len(y),)
    else:
        sample_weight /= np.sum(sample_weight)

    if X.dtype in ("float", "float32"):
        output = c_classifiers.train_regression_stump_double(
            X.T,
            np.array(y.flatten(), dtype="float"),
            sample_weight,
            argsorted_X if argsorted_X is not None else np.argsort(X.T, axis=1),
        )
    else:
        output = c_classifiers.train_regression_stump_int32(
            X.T,
            np.array(y.flatten(), dtype="float"),
            sample_weight,
            argsorted_X if argsorted_X is not None else np.argsort(X.T, axis=0),
        )

    return {
        "min_value": float(output[0]),
        "best_dim": int(output[1]),
        "threshold": float(output[2]),
        "coefficient": float(output[3]),
        "constant": float(output[4]),
    }


def _fit_regressor_stump_c_ext_threaded(X, y, sample_weight, argsorted_X=None):
    if c_classifiers is None:
        return _fit_regressor_stump_threaded(X, y, sample_weight, argsorted_X)

    Y = y.flatten()

    if sample_weight is None:
        sample_weight = np.ones(shape=(X.shape[0],), dtype="float") / (X.shape[0],)
    else:
        sample_weight /= np.sum(sample_weight)

    classifier_result = []

    with cfut.ThreadPoolExecutor(max_workers=psutil.cpu_count()) as tpe:
        futures = []
        if argsorted_X is not None:
            for dim in range(X.shape[1]):
                futures.append(
                    tpe.submit(_regressor_c_learn_one_dimension, dim, X[:, dim], Y, sample_weight, argsorted_X[:, dim])
                )
        else:
            for dim in range(X.shape[1]):
                futures.append(tpe.submit(_regressor_c_learn_one_dimension, dim, X[:, dim], Y, sample_weight))
        for future in cfut.as_completed(futures):
            classifier_result.append(future.result())

    # Sort the returned data after lowest error.
    classifier_result = sorted(classifier_result, key=itemgetter(1))
    best_result = classifier_result[0]

    return {
        "best_dim": int(best_result[0]),
        "min_value": float(best_result[1]),
        "threshold": float(best_result[2]),
        "coefficient": float(best_result[3]),
        "constant": float(best_result[4]),
    }


def _regressor_learn_one_dimension(dim_nbr, x, y, sample_weights, sorting_argument=None):
    n_samples = len(x)
    if sorting_argument is None:
        sorting_argument = np.argsort(x)

    # Sort the weights and labels with argument for this dimension.
    # Time: 25%
    sorted_weights = sample_weights[sorting_argument]
    sorted_output = y[sorting_argument]

    # Cumulative sum of desired output multiplied with weights.
    # Time: 10 %
    Szw = (sorted_weights * sorted_output).cumsum()
    # Cumulative sum of the weights.
    Sw = sorted_weights.cumsum()

    # Calculate regression function parameters.
    # Time: 25 %
    b = Szw / Sw
    zz = np.where((1.0 - Sw) < 1e-14)
    Sw[zz] = 0.0
    a = ((Szw[-1] - Szw) / (1 - Sw)) - b
    Sw[zz] = 1.0

    # Calculate the weighted square error:
    # Time: 40 %
    e = (
        (sorted_weights * (sorted_output * sorted_output)).sum()
        - (2 * a * (Szw[-1] - Szw))
        - (2 * b * Szw[-1])
        + ((a * a) + (2 * a * b)) * (1 - Sw)
        + (b * b)
    )

    del sorted_output
    del sorted_weights
    del Szw
    del Sw

    min_ind = e.argmin()
    error = e[min_ind]
    del e
    coeff = a[min_ind]
    del a
    constant = b[min_ind]
    del b

    # Handle floating point data different from integer data when it comes
    # to setting the threshold.
    if x.dtype in ("float", "float32"):
        if min_ind == (n_samples - 1):
            threshold = x[sorting_argument[min_ind]] + 0.1
        elif min_ind == 0:
            threshold = x[sorting_argument[min_ind]] - 0.1
        else:
            threshold = (x[sorting_argument[min_ind]] + x[sorting_argument[min_ind + 1]]) / 2
    else:
        if min_ind == (n_samples - 1):
            threshold = np.floor(x[sorting_argument[min_ind]]) + 1
        elif min_ind == 0:
            threshold = np.floor(x[sorting_argument[min_ind]]) - 1
        else:
            v1 = int(x[sorting_argument[min_ind]])
            v2 = int(x[sorting_argument[min_ind + 1]])
            thr = (v1 + v2) / 2
            if np.abs(thr) > (2**31):
                print("Threshold for dimension {0} was greater than 32 bit integer!".format(dim_nbr))
            threshold = np.int32(thr)

    del sorting_argument

    return dim_nbr, error, threshold, coeff, constant


def _regressor_c_learn_one_dimension(dim_nbr, x, y, sample_weights, sorting_argument=None):

    output = c_classifiers.fit_regression_stump(
        x,
        np.array(y, dtype="float"),
        sample_weights,
        sorting_argument if sorting_argument is not None else np.argsort(x),
    )

    return [
        dim_nbr,
    ] + list(output)
