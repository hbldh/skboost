#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`nor`
==================

.. module:: nor
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-05

"""


import numpy as np

from skboost.milboost.softmax import SoftmaxFunction


class NoisyOR(SoftmaxFunction):
    """The Noisy OR softmax function."""

    def __init__(self, *args):
        super(NoisyOR, self).__init__(*args)

    def f(self, x):
        """

        :param x: The values to evaluate softmax over.
        :type x: :py:class:`numpy.ndarray`
        :return: The NOR softmax value.
        :rtype: float

        """
        return 1 - np.prod(1 - x)

    def d_dt(self, x):
        return (1 - self.f(x)) / (1 - x)
