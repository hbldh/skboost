#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`run`
==================

.. module:: run
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-06, 14:24

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import zero_one_loss
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from skboost.gentleboost import GentleBoostClassifier
from skboost.datasets import MUSK1, MUSK2, Hastie_10_2
from skboost.milboost.softmax import *

ds_train = MUSK2()
print(ds_train)
ds_test = MUSK1()
print(ds_test)
#ds_train = Hastie_10_2()
#ds_test = Hastie_10_2(random_state=1)


classifier = GentleBoostClassifier(
    base_estimator=DecisionTreeRegressor(max_depth=1),
    n_estimators=20,
    learning_rate=1.0)
classifier.fit(ds_train.data, np.sign(ds_train.labels))
print(classifier)
print(classifier.predict_proba(ds_test.data))

# Calculate step-by-step results per instance on training data.
milboost_train_error = np.zeros((len(classifier.estimators_),))
for i, y_pred in enumerate(classifier.staged_predict(ds_train.data)):
    milboost_train_error[i] = zero_one_loss(y_pred, np.sign(ds_train.labels))

# Calculate step-by-step results per instance on test data.
milboost_test_error = np.zeros((len(classifier.estimators_),))
for i, y_pred in enumerate(classifier.staged_predict(ds_test.data)):
    milboost_test_error[i] = zero_one_loss(y_pred, np.sign(ds_test.labels))

# Calculate step-by-step results per bag on training data.
milboost_bag_train_error = np.zeros((len(classifier.estimators_),))
for i, y_pred in enumerate(classifier.staged_predict(ds_train.data)):
    bag_labels_pred = np.array([np.max(x) for x in np.split(y_pred, ds_train.bag_partitioning)], 'int')
    milboost_bag_train_error[i] = zero_one_loss(bag_labels_pred, ds_train.bag_labels)

milboost_bag_test_error = np.zeros((len(classifier.estimators_),))
for i, y_pred in enumerate(classifier.staged_predict(ds_test.data)):
    bag_labels_pred = np.array([np.max(x) for x in np.split(y_pred, ds_test.bag_partitioning)], 'int')
    milboost_bag_test_error[i] = zero_one_loss(bag_labels_pred, np.sign(ds_test.bag_labels))


plt.plot(np.arange(len(classifier.estimators_)) + 1, milboost_test_error, '{0}'.format('r'),
         label='{0} Test Error'.format('GentleBoost'))
plt.plot(np.arange(len(classifier.estimators_)) + 1, milboost_train_error, '{0}--'.format('g'),
         label='{0} Train Error'.format('GentleBoost'))
plt.plot(np.arange(len(classifier.estimators_)) + 1, milboost_bag_test_error, '{0}'.format('b'),
         label='{0} Bag Test Error'.format('GentleBoost'))
plt.plot(np.arange(len(classifier.estimators_)) + 1, milboost_bag_train_error, '{0}--'.format('c'),
         label='{0} Bag Train Error'.format('GentleBoost'))

plt.gca().set_ylim((0.0, 0.5))
plt.xlabel('n_estimators')
plt.ylabel('error rate')
plt.legend(loc='upper right', fancybox=True)

plt.show()
