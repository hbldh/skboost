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

from nose import tools
import numpy as np
from numpy.testing import *

from skboost.datasets import MUSK1, Hastie_10_2
from skboost.gentleboost import GentleBoostClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import zero_one_loss


class TestGentleBoost(object):

    def setup(self):
        self.data_1 = MUSK1()
        self.data_2 = Hastie_10_2()

    def test_musk_fitting(self):
        c = GentleBoostClassifier(
            base_estimator=DecisionTreeRegressor(max_depth=1),
            n_estimators=30,
            learning_rate=1.0
        )
        c.fit(self.data_1.data, np.sign(self.data_1.labels))
        assert_array_less(c.estimator_errors_, 0.5)
        assert zero_one_loss(np.sign(self.data_1.labels), c.predict(self.data_1.data)) < 0.1

    def test_hastie_fitting(self):
        c = GentleBoostClassifier(
            base_estimator=DecisionTreeRegressor(max_depth=1),
            n_estimators=30,
            learning_rate=1.0
        )
        c.fit(self.data_2.data, np.sign(self.data_2.labels))
        assert_array_less(c.estimator_errors_, 0.5)
        assert zero_one_loss(np.sign(self.data_2.labels), c.predict(self.data_2.data)) < 0.2
