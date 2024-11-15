#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`isr`
==================

.. module:: isr
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-05

"""


import numpy as np

from skboost.milboost.softmax import SoftmaxFunction


class ISR(SoftmaxFunction):
    """The ISR softmax function."""

    def __init__(self, *args):
        super(ISR, self).__init__(*args)

    def f(self, x):
        # FIXME: Breaks when x_i == 1.0
        x[x == 1.0] -= 1e-10
        s = np.sum(x / (1 - x))
        return s / (1 + s)

    def f2(self, x):
        s = np.sum(np.exp(x))
        return s / (1 + s)

    def d_dt(self, x):
        return ((1 - self.f(x)) / (1 - x)) ** 2
