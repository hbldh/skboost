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

from nose import tools
import numpy as np
from numpy.testing import *

from skboost.datasets import MUSK1, Hastie_10_2
from skboost.milboost import *
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import zero_one_loss


class TestMILBoost(object):

    def setup(self):
        self.data_1 = MUSK1()
        self.data_2 = Hastie_10_2()

    def test_musk_fitting_lse(self):
        c = MILBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            softmax=LogSumExponential(5.0),
            n_estimators=30,
            learning_rate=1.0
        )
        c.fit(self.data_1.data, self.data_1.labels)
        assert_array_less(c.estimator_errors_, 0.5)
        assert zero_one_loss(np.sign(self.data_1.labels), c.predict(self.data_1.data)) < 0.30

    def test_hastie_fitting(self):
        c = MILBoostClassifier(
            base_estimator=DecisionTreeClassifier(max_depth=1),
            softmax=LogSumExponential(5.0),
            n_estimators=30,
            learning_rate=1.0
        )
        c.fit(self.data_2.data, self.data_2.labels)
        assert_array_less(c.estimator_errors_, 0.5)
        assert zero_one_loss(np.sign(self.data_2.labels), c.predict(self.data_2.data)) < 0.40
