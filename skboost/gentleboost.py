#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`gentleboost`
==================

.. module:: gentleboost
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2014-08-30, 22:25

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np

from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble.weight_boosting import BaseWeightBoosting, ClassifierMixin, RegressorMixin, DecisionTreeRegressor


class GentleBoostClassifier(BaseWeightBoosting, ClassifierMixin):
    """An implementation of the GentleBoost classifier.

    The Gentle AdaBoost algorithm (population version) uses Newton steps for
    minimizing E[exp(−yF(x))]. It is a modified version of the Real AdaBoost algorithm,
    using Newton stepping rather than exact optimization at each step.

    """

    def __init__(self,
                 base_estimator=DecisionTreeRegressor(max_depth=1),
                 n_estimators=100,
                 learning_rate=1.,
                 random_state=None):
        super(GentleBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self.threshold = 0.0

    def fit(self, X, y, sample_weight=None):
        """Build a boosted classifier from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like of shape = [n_samples]
            The target values (integers that correspond to classes).

        sample_weight : array-like of shape = [n_samples], optional
            Sample weights. If None, the sample weights are initialized to
            ``1 / n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """
        # Check that the base estimator is a classifier
        if not isinstance(self.base_estimator, RegressorMixin):
            raise TypeError("base_estimator must be a subclass of RegressorMixin")

        n_classes = len(np.unique(y))
        if n_classes != 2:
            raise ValueError("The GentleBoost classifier is a binary classifier, "
                             "and the labels array had {0} classes.".format(n_classes))

        return super(GentleBoostClassifier, self).fit(X, y, sample_weight)

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Perform a single boost according to the real multi-class SAMME.R
        algorithm or to the discrete SAMME algorithm and return the updated
        sample weights.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : array-like of shape = [n_samples, n_features]
            The training input samples.

        y : array-like of shape = [n_samples]
            The target values (integers that correspond to classes).

        sample_weight : array-like of shape = [n_samples]
            The current sample weights.

        random_state : numpy.RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape = [n_samples] or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        estimator_error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        estimator = self._make_estimator(append=False)

        try:
            estimator.set_params(random_state=self.random_state)
        except ValueError:
            pass

        fitted_estimator = estimator.fit(X, y, sample_weight=sample_weight)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = 2

        y_predict = fitted_estimator.predict(X)

        # Instances incorrectly classified
        incorrect = np.sign(y_predict) != y
        # Error fraction
        estimator_error = np.mean(
            np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            # Only boost positive weights
            sample_weight *= np.exp(-(y * y_predict))

        self.estimators_.append(fitted_estimator)

        return sample_weight, 1., estimator_error

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes.
        """
        return ((self.decision_function(X) > self.threshold) * 2) - 1

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
        for pred in self.staged_decision_function(X):
            yield ((pred > self.threshold) * 2) - 1

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

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
        X = np.asarray(X)

        return np.sum(estimator.predict(X) for estimator in self.estimators_)

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

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
        X = np.asarray(X)

        pred = None

        for estimator in self.estimators_:
            # The weights are all 1.0 for GentleBoost
            if pred is None:
                pred = estimator.predict(X)
            else:
                pred += estimator.predict(X)

            yield pred

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        X = np.asarray(X)

        return 1.0 / (1 + np.exp(-(self.decision_function(X) - self.threshold)))

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
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : generator of array, shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        X = np.asarray(X)
        for dv in self.staged_decision_function(X):
            yield 1.0 / (1 + np.exp(-(dv - self.threshold)))

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        return np.log(self.predict_proba(X))
