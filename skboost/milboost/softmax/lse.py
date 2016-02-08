#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`lse`
==================

.. module:: lse
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@swedwise.com>

Created on 2015-11-05, 16:30

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np

from skboost.milboost.softmax import SoftmaxFunction


class LogSumExponential(SoftmaxFunction):
    """The Log-Sum_Exponential softmax function.

    https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/

    :param x: The values to evaluate softmax over.
    :type x: :py:class:`numpy.ndarray`
    :param r: The radius of the LSE. Defaults to 1.0
    :type r: float
    :return: The LSE softmax value.
    :rtype: float

    """

    def __str__(self):
        return super(LogSumExponential, self).__str__() + "({0:.1f})".format(self.radius)

    def __init__(self, *args):
        super(LogSumExponential, self).__init__(*args)
        self.radius = float(args[0]) if len(args) > 0 else 1.0

    def f(self, x):
        shift = np.max(x)
        return shift + (np.log(np.sum(np.exp(self.radius * (x - shift))) / len(x)) / self.radius)

    def d_dt(self, x):
        return np.exp(self.radius * x) / np.sum(np.exp(self.radius * x))
