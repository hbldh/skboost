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

import pytest

import numpy as np
from numpy.testing import assert_array_less

from skboost.milboost.softmax import LogSumExponential
from skboost.milboost.softmax import ISR
from skboost.milboost.softmax import GeneralizedMean
from skboost.milboost.softmax import NoisyOR

@pytest.fixture(scope='module')
def t():
    t = np.linspace(0.0, 1.0, num=100)
    stacked_t = np.vstack([t, 1 - t]).T
    return stacked_t


@pytest.fixture(scope='module')
def true_t():
    t = np.linspace(0.0, 1.0, num=100)
    stacked_t = np.vstack([t, 1 - t]).T
    true_t = np.max(stacked_t, axis=1)
    return true_t


def test_lse_1():
    lse = LogSumExponential(5.0)
    lse_vals = [lse.f(x) for x in t()]
    assert_array_less(lse_vals, true_t())


def test_lse_2():
    lse = LogSumExponential(20.0)
    lse_vals = [lse.f(x) for x in t()]
    assert_array_less(lse_vals, true_t())


def test_lse_3():
    lse = LogSumExponential()
    assert lse.radius == 1.0
    lse_vals = [lse.f(x) for x in t()]
    assert_array_less(lse_vals, true_t())


def test_lse_4():
    lse_vals_1 = np.array([LogSumExponential(1.0).f(x) for x in t()])
    lse_vals_5 = np.array([LogSumExponential(5.0).f(x) for x in t()])
    lse_vals_20 = np.array([LogSumExponential(20.0).f(x) for x in t()])
    assert_array_less(lse_vals_1, lse_vals_5)
    assert_array_less(lse_vals_5, lse_vals_20)


def test_nor_1():
    nor = NoisyOR()
    nor_vals = [nor.f(x) for x in t()]
    assert np.all((true_t() - nor_vals) <= 0.0)


def test_isr_1():
    isr = ISR()
    isr_vals = [isr.f(x) for x in t()]
    assert np.all((true_t()[1:-1] - isr_vals[1:-1]) <= 0.0)


def test_gm_1():
    gm = GeneralizedMean(5.0)
    gm_vals = [gm.f(x) for x in t()]
    assert_array_less(gm_vals, true_t())


def test_gm_2():
    gm = GeneralizedMean(20.0)
    gm_vals = [gm.f(x) for x in t()]
    assert_array_less(gm_vals, true_t())


def test_gm_3():
    gm = GeneralizedMean()
    assert gm.radius == 1.0
    gm_vals = [gm.f(x) for x in t()]
    assert_array_less(gm_vals, true_t())


def test_gm_4():
    gm_vals_1 = np.array([GeneralizedMean(1.0).f(x) for x in t()])
    gm_vals_5 = np.array([GeneralizedMean(5.0).f(x) for x in t()])
    gm_vals_20 = np.array([GeneralizedMean(20.0).f(x) for x in t()])
    assert_array_less(gm_vals_1, gm_vals_5)
    assert_array_less(gm_vals_5, gm_vals_20)
