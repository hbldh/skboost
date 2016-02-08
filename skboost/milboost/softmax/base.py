#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`base`
==================

.. module:: base
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-06

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


class SoftmaxFunction(object):
    """The Generalized Mean softmax function."""

    def __init__(self, *args):
        pass

    def __str__(self):
        return "{0}".format(self.__class__.__name__)

    def __repr__(self):
        return str(self)

    def __call__(self, x):
        return self.f(x)

    def f(self, x):
        raise NotImplementedError()

    def d_dt(self, x):
        raise NotImplementedError()
