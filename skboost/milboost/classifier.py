#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`milboost`
==================

.. module:: milboost
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-06, 08:48

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import warnings

import numpy as np
from sklearn.ensemble.weight_boosting import ClassifierMixin, BaseWeightBoosting
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import fminbound

from skboost.milboost.softmax import SoftmaxFunction

__all__ = ['MILBoostClassifier', ]


class MILBoostClassifier(ClassifierMixin, BaseWeightBoosting):

    def __init__(self,
                 base_estimator=DecisionTreeClassifier(max_depth=10),
                 softmax=None,
                 n_estimators=50,
                 learning_rate=1.0,
                 random_state=None,
                 verbose=False):

        super(MILBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        if not isinstance(softmax, SoftmaxFunction):
            raise TypeError("Softmax input must be an object of class `SoftmaxFunction`")
        self.softmax_fcn = softmax
        self._verbose = verbose

        self._bag_labels = None
        self._inferred_y = None
        self._bag_partitioning = None

    def __str__(self):
        return "{0}, with {1} {2} classifiers".format(
            self.__class__.__name__, len(self.estimators_), self.estimators_[0])

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        y : array-like of shape = [n_samples]
            The target values (class labels).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """

        # Pre-compute bag labels and inferred instance labels from y.
        unique_bag_ids = np.unique(y)
        self._bag_labels = np.zeros((max(np.abs(unique_bag_ids)) + 1, ), 'int')
        self._bag_labels[np.abs(unique_bag_ids)] = np.sign(unique_bag_ids)
        self._bag_labels = self._bag_labels[1:]
        self._inferred_y = np.sign(y)
        self._bag_partitioning = np.cumsum(np.bincount(np.abs(y))[1:])

        # Fit
        out = super(MILBoostClassifier, self).fit(X, y, sample_weight)

        # Clean away stored labels.
        self._bag_labels = None
        self._inferred_y = None
        self._bag_partitioning = None

        return out

    def _boost(self, iboost, X, y, sample_weight):

        if iboost > 0:
            dv_pre = self.decision_function(X)
            instance_probabilites = self._estimate_instance_probabilities(dv_pre)
            bag_probabilites = self._estimate_bag_probabilites(instance_probabilites)
            sample_weight = self._calculate_new_weights(instance_probabilites, bag_probabilites)
        else:
            dv_pre = np.zeros(((X.shape[0]),), 'float')

        estimator = self._make_estimator()
        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        _weights = np.abs(sample_weight)
        estimator.fit(X, self._inferred_y, sample_weight=_weights)
        y_predict = estimator.predict(X)

        # Instances incorrectly classified
        incorrect = y_predict != self._inferred_y

        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=_weights, axis=0))

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        # Estimate alpha, the estimator weight.
        estimator_weight, nll = self._find_estimator_weight(y, dv_pre, y_predict)
        if self._verbose:
            print("[{0}] - err={1:.4f}, w={2:.4f}, -L={3:.4f}".format(iboost, estimator_error, estimator_weight, nll))

        if estimator_weight < 1e-5:
            _weights = None

        return _weights, estimator_weight, estimator_error

    def _negative_log_likelihood(self, bag_probabilities):
        positive_bags_log_prob = np.log(bag_probabilities[self._bag_labels > 0])
        positive_bags_log_prob[np.isinf(positive_bags_log_prob)] = 0.0
        negative_bags_log_prob = np.log(1 - bag_probabilities[self._bag_labels < 0])
        negative_bags_log_prob[np.isinf(negative_bags_log_prob)] = 0.0

        return -(np.sum(positive_bags_log_prob) + np.sum(negative_bags_log_prob))

    def _find_estimator_weight(self, y, dv_pre, y_pred):
        """Make line search to determine estimator weights."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            def optimization_function(alpha):
                p_ij = self._estimate_instance_probabilities(dv_pre + alpha * y_pred)
                p_i = self._estimate_bag_probabilites(p_ij)
                return self._negative_log_likelihood(p_i)

            # TODO: Add option to choose optimization method.

            alpha, fval, err, n_func = fminbound(optimization_function, 0.0, 5.0, full_output=True, disp=1)
            if self.learning_rate < 1.0:
                alpha *= self.learning_rate
        return alpha, fval

    def _estimate_instance_probabilities(self, dv):
        return 1.0 / (1 + np.exp(-(2 * dv)))

    def _estimate_bag_probabilites(self, instance_probabilites):
        bags = self._bag_split(instance_probabilites)
        bag_probabilities = np.array([self.softmax_fcn.f(x) for x in bags])
        return bag_probabilities

    def _calculate_new_weights(self, instance_probabilites, bag_probabilities):
        weights = []
        for p_ij, p_i, Y_i in zip(self._bag_split(instance_probabilites),
                                  bag_probabilities,
                                  self._bag_labels):
            if Y_i > 0:
                if p_i == 0.0:
                    p_i = np.finfo(float).resolution
                term_1 = (2 * p_ij * (1 - p_ij)) / p_i
            else:
                if p_i == 1.0:
                    p_i = 1 - np.finfo(float).resolution
                term_1 = -((2 * p_ij * (1 - p_ij)) / (1 - p_i))
            weights += (term_1 * self.softmax_fcn.d_dt(p_ij)).tolist()

        return np.array(weights) / np.sum(np.abs(weights))

    def _bag_split(self, x):
        return np.split(x, self._bag_partitioning)[:-1]

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        score : array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        classes = self.classes_[:, np.newaxis]
        pred = sum((estimator.predict(X) == classes).T * w
                   for estimator, w in zip(self.estimators_,
                                           self.estimator_weights_))
        pred[:, 0] *= -1
        return pred.sum(axis=1)

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        score : generator of array, shape = [n_samples, k]
            The decision function of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self, "n_classes_")
        X = self._validate_X_predict(X)

        classes = self.classes_[:, np.newaxis]
        pred = None

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):

            current_pred = estimator.predict(X)
            current_pred = (current_pred == classes).T * weight

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

            tmp_pred = np.copy(pred)
            tmp_pred[:, 0] *= -1
            yield (tmp_pred).sum(axis=1)

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        pred = self.decision_function(X)

        return self.classes_.take(pred > 0, axis=0)

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : generator of array, shape = [n_samples]
            The predicted classes.
        """
        classes = self.classes_

        for pred in self.staged_decision_function(X):
            yield np.array(classes.take(pred > 0, axis=0))

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        check_is_fitted(self, "n_classes_")

        n_classes = self.n_classes_
        X = self._validate_X_predict(X)

        proba = sum(estimator.predict_proba(X) * w
                    for estimator, w in zip(self.estimators_,
                                            self.estimator_weights_))

        proba /= self.estimator_weights_.sum()
        proba = np.exp((1. / (n_classes - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba /= normalizer

        return proba

    def staged_predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : generator of array, shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        X = self._validate_X_predict(X)

        n_classes = self.n_classes_
        proba = None
        norm = 0.

        for weight, estimator in zip(self.estimator_weights_,
                                     self.estimators_):
            norm += weight

            current_proba = estimator.predict_proba(X) * weight

            if proba is None:
                proba = current_proba
            else:
                proba += current_proba

            real_proba = np.exp((1. / (n_classes - 1)) * (proba / norm))
            normalizer = real_proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            real_proba /= normalizer

            yield real_proba

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape = [n_samples, n_features]
            The training input samples. Sparse matrix can be CSC, CSR, COO,
            DOK, or LIL. DOK and LIL are converted to CSR.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        return np.log(self.predict_proba(X))
