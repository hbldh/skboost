#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`logitboost`
==================

.. module:: logitboost
   :platform: Unix, Windows
   :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2014-09-12, 22:57

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np

from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble.weight_boosting import DecisionTreeRegressor
from sklearn.ensemble.weight_boosting import BaseWeightBoosting, ClassifierMixin, RegressorMixin


class LogitBoostClassifier(BaseWeightBoosting, ClassifierMixin):
    """An implementation of the LogitBoost classifier.

    LogitBoost is an adaptive Newton algorithm for ﬁtting an
    additive logistic regression model  by stagewise optimization of
    the Bernoulli log-likelihood.

    """

    def __init__(self,
                 base_estimator=DecisionTreeRegressor(max_depth=1),
                 n_estimators=50,
                 learning_rate=1.,
                 random_state=None):
        super(LogitBoostClassifier, self).__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state)

        self._ensemble_classifier_response = None
        self._instance_probabilities = None
        self._01_labels = None

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

        if len(np.unique(y)) != 2:
            raise ValueError("Only binary classification LogitBoost is implemented as of yet.")

        out = super(LogitBoostClassifier, self).fit(X, y, sample_weight)

        self._ensemble_classifier_response = None
        self._instance_probabilities = None
        self._01_labels = None

        return out

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost iteration.

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

        if iboost == 0:
            self._ensemble_classifier_response = np.zeros((len(y), ), dtype='float')
            self._instance_probabilities = np.ones((len(y), ), dtype='float') / 2
            self._01_labels = (y + 1) / 2

        # Update the weights
        sample_weight = (self._instance_probabilities * (1 - self._instance_probabilities))
        # Calculate the working response.
        z = (self._01_labels - self._instance_probabilities) / sample_weight

        # When instance probabilities == 1, the weights are set to zero, creating division
        # by zero NaNs in the z array. This part sets these to the minimal value they can
        # be set to during the fitting to represent that we are very sure of these.
        # The warnings catcher suppresses the printout of these warnings.
        z_nans = np.isnan(z) | np.isinf(z)
        z[z_nans] = self._01_labels[z_nans]

        fitted_estimator = estimator.fit(X, z, sample_weight=sample_weight)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = 2

        y_predict = fitted_estimator.predict(X)
        self._ensemble_classifier_response += 0.5 * y_predict

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
            exp_F = np.exp(self._ensemble_classifier_response)
            self._instance_probabilities = exp_F / (exp_F + np.exp(-self._ensemble_classifier_response))

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

        pred = None

        for estimator in self.estimators_:
            # The weights are all 1. for LogitBoost
            current_pred = estimator.predict(X)

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

        return pred

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
            # The weights are all 1. for GentleBoost
            current_pred = estimator.predict(X)

            if pred is None:
                pred = current_pred
            else:
                pred += current_pred

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
        n_classes = self.n_classes_

        if n_classes > 2:
            raise NotImplementedError()
        else:
            proba = 1.0 / (1 + np.exp(-(self.decision_function(X) - self.threshold)))

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
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : generator of array, shape = [n_samples]
            The class probabilities of the input samples. The order of
            outputs is the same of that of the `classes_` attribute.
        """
        n_classes = self.n_classes_

        for dv in self.staged_decision_function(X):
            if n_classes > 2:
                raise NotImplementedError()
            else:
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
