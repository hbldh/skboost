#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`test_softmax`
==================

.. module:: test_softmax
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-05

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from numpy.testing import assert_array_less

from skboost.milboost.softmax import LogSumExponential
from skboost.milboost.softmax import ISR
from skboost.milboost.softmax import GeneralizedMean
from skboost.milboost.softmax import NoisyOR


class TestLSE(object):

    def setup(self):
        t = np.linspace(0.0, 1.0, num=100)
        self.t = np.vstack([t, 1 - t]).T
        self.true_t = np.max(self.t, axis=1)

    def test_lse_1(self):
        lse = LogSumExponential(5.0)
        lse_vals = map(lambda x: lse.f(x), self.t)
        assert_array_less(lse_vals, self.true_t)

    def test_lse_2(self):
        lse = LogSumExponential(20.0)
        lse_vals = map(lse.f, self.t)
        assert_array_less(lse_vals, self.true_t)

    def test_lse_3(self):
        lse = LogSumExponential()
        assert lse.radius == 1.0
        lse_vals = map(lse.f, self.t)
        assert_array_less(lse_vals, self.true_t)

    def test_lse_4(self):
        lse_vals_1 = np.array(map(LogSumExponential(1.0).f, self.t))
        lse_vals_5 = np.array(map(LogSumExponential(5.0).f, self.t))
        lse_vals_20 = np.array(map(LogSumExponential(20.0).f, self.t))
        assert_array_less(lse_vals_1, lse_vals_5)
        assert_array_less(lse_vals_5, lse_vals_20)


class TestNOR(object):

    def setup(self):
        t = np.linspace(0.0, 1.0, num=100)
        self.t = np.vstack([t, 1 - t]).T
        self.true_t = np.max(self.t, axis=1)

    def test_nor_1(self):
        nor = NoisyOR()
        nor_vals = map(nor.f, self.t)
        assert np.all((self.true_t - nor_vals) <= 0.0)


class TestISR(object):

    def setup(self):
        t = np.linspace(0.0, 1.0, num=100)
        self.t = np.vstack([t, 1 - t]).T
        self.true_t = np.max(self.t, axis=1)

    def test_isr_1(self):
        isr = ISR()
        isr_vals = map(isr.f, self.t)
        assert np.all((self.true_t[1:-1] - isr_vals[1:-1]) <= 0.0)


class TestGM(object):

    def setup(self):
        t = np.linspace(0.0, 1.0, num=100)
        self.t = np.vstack([t, 1 - t]).T
        self.true_t = np.max(self.t, axis=1)

    def test_gm_1(self):
        gm = GeneralizedMean(5.0)
        gm_vals = map(lambda x: gm.f(x), self.t)
        assert_array_less(gm_vals, self.true_t)

    def test_gm_2(self):
        gm = GeneralizedMean(20.0)
        gm_vals = map(gm.f, self.t)
        assert_array_less(gm_vals, self.true_t)

    def test_gm_3(self):
        gm = GeneralizedMean()
        assert gm.radius == 1.0
        gm_vals = map(gm.f, self.t)
        assert_array_less(gm_vals, self.true_t)

    def test_gm_4(self):
        gm_vals_1 = np.array(map(GeneralizedMean(1.0).f, self.t))
        gm_vals_5 = np.array(map(GeneralizedMean(5.0).f, self.t))
        gm_vals_20 = np.array(map(GeneralizedMean(20.0).f, self.t))
        assert_array_less(gm_vals_1, gm_vals_5)
        assert_array_less(gm_vals_5, gm_vals_20)
