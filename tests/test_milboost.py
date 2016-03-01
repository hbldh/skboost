#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`test_milboost`
==================

.. module:: test_milboost
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-12

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from numpy.testing import *

from skboost.datasets import MUSK1, Hastie_10_2
from skboost.milboost import *
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import zero_one_loss


def test_milboost_musk_fitting_lse():
    c = MILBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        softmax=LogSumExponential(5.0),
        n_estimators=30,
        learning_rate=1.0
    )

    data = MUSK1()
    c.fit(data.data, data.labels)
    assert_array_less(c.estimator_errors_, 0.5)
    assert zero_one_loss(np.sign(data.labels), c.predict(data.data)) < 0.30


def test_milboost_hastie_fitting():
    c = MILBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        softmax=LogSumExponential(5.0),
        n_estimators=30,
        learning_rate=1.0
    )

    data = Hastie_10_2()
    c.fit(data.data, data.labels)
    assert_array_less(c.estimator_errors_, 0.5)
    assert zero_one_loss(np.sign(data.labels), c.predict(data.data)) < 0.40
