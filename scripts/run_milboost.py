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
from sklearn.tree import DecisionTreeClassifier

from skboost.milboost.classifier import MILBoostClassifier
from skboost.datasets import MUSK1, MUSK2, Hastie_10_2
from skboost.milboost.softmax import *

#ds_train = MUSK2()
#print(ds_train)
#ds_test = MUSK1()
#print(ds_test)
ds_train = Hastie_10_2()
ds_test = Hastie_10_2(random_state=54325)



mil_classifier = MILBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    #softmax=ISR(),
    #softmax=GeneralizedMean(5.0),
    #softmax=NoisyOR(),
    softmax=LogSumExponential(5.0),
    n_estimators=30,
    learning_rate=1.0,
    verbose=True)
mil_classifier.fit(ds_train.data, ds_train.labels)
print(mil_classifier)

# Calculate step-by-step results per instance on training data.
milboost_train_error = np.zeros((len(mil_classifier.estimators_),))
for i, y_pred in enumerate(mil_classifier.staged_predict(ds_train.data)):
    milboost_train_error[i] = zero_one_loss(y_pred, np.sign(ds_train.labels))

# Calculate step-by-step results per instance on test data.
milboost_test_error = np.zeros((len(mil_classifier.estimators_),))
for i, y_pred in enumerate(mil_classifier.staged_predict(ds_test.data)):
    milboost_test_error[i] = zero_one_loss(y_pred, np.sign(ds_test.labels))

# Calculate step-by-step results per bag on training data.
milboost_bag_train_error = np.zeros((len(mil_classifier.estimators_),))
for i, y_pred in enumerate(mil_classifier.staged_predict(ds_train.data)):
    bag_labels_pred = np.array([np.max(x) for x in np.split(y_pred, ds_train.bag_partitioning)], 'int')
    milboost_bag_train_error[i] = zero_one_loss(bag_labels_pred, ds_train.bag_labels)

milboost_bag_test_error = np.zeros((len(mil_classifier.estimators_),))
for i, y_pred in enumerate(mil_classifier.staged_predict(ds_test.data)):
    bag_labels_pred = np.array([np.max(x) for x in np.split(y_pred, ds_test.bag_partitioning)], 'int')
    milboost_bag_test_error[i] = zero_one_loss(bag_labels_pred, np.sign(ds_test.bag_labels))


plt.plot(np.arange(len(mil_classifier.estimators_)) + 1, milboost_test_error, '{0}'.format('r'),
         label='{0} Test Error'.format('MILBoost, {0}'.format(mil_classifier.softmax_fcn)))
plt.plot(np.arange(len(mil_classifier.estimators_)) + 1, milboost_train_error, '{0}--'.format('g'),
         label='{0} Train Error'.format('MILBoost, {0}'.format(mil_classifier.softmax_fcn)))
plt.plot(np.arange(len(mil_classifier.estimators_)) + 1, milboost_bag_test_error, '{0}'.format('b'),
         label='{0} Bag Test Error'.format('MILBoost, {0}'.format(mil_classifier.softmax_fcn)))
plt.plot(np.arange(len(mil_classifier.estimators_)) + 1, milboost_bag_train_error, '{0}--'.format('c'),
         label='{0} Bag Train Error'.format('MILBoost, {0}'.format(mil_classifier.softmax_fcn)))

plt.gca().set_ylim((0.0, 0.5))
plt.xlabel('n_estimators')
plt.ylabel('error rate')
plt.legend(loc='upper right', fancybox=True)

plt.show()
