#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`gm`
==================

.. module:: gm
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

from skboost.milboost.softmax import SoftmaxFunction


class GeneralizedMean(SoftmaxFunction):
    """The Generalized Mean softmax function."""

    def __init__(self, *args):
        super(GeneralizedMean, self).__init__(*args)
        self.radius = float(args[0]) if len(args) > 0 else 1.0

    def __str__(self):
        return super(GeneralizedMean, self).__str__() + "({0:.1f})".format(self.radius)

    def f(self, x):
        return (np.sum(x ** self.radius) / len(x)) ** (1 / self.radius)

    def d_dt(self, x):
        return self.f(x) * ((x ** (self.radius - 1.0)) / np.sum(x ** self.radius))
