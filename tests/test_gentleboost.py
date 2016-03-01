#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`test_gentleboost`
==================

.. module:: test_gentleboost
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-11

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_array_less

from skboost.datasets import MUSK1, Hastie_10_2
from skboost.gentleboost import GentleBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import zero_one_loss


def test_gentleboost_musk_fitting():
    c = GentleBoostClassifier(
        base_estimator=DecisionTreeRegressor(max_depth=1),
        n_estimators=30,
        learning_rate=1.0
    )
    data = MUSK1()
    c.fit(data.data, np.sign(data.labels))
    assert_array_less(c.estimator_errors_, 0.5)
    assert zero_one_loss(np.sign(data.labels), c.predict(data.data)) < 0.1


def test_gentleboost_hastie_fitting():
    c = GentleBoostClassifier(
        base_estimator=DecisionTreeRegressor(max_depth=1),
        n_estimators=30,
        learning_rate=1.0
    )
    data = Hastie_10_2()
    c.fit(data.data, np.sign(data.labels))
    assert_array_less(c.estimator_errors_, 0.5)
    assert zero_one_loss(np.sign(data.labels), c.predict(data.data)) < 0.2
