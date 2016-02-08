#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`hastie`
===========

.. module:: hastie
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-10

"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import six
import numpy as np
from sklearn.datasets import make_hastie_10_2

__all__ = ['Hastie_10_2', ]


class Hastie_10_2(object):
    """Returns a Hastie et al. 2009, Example 10.2 dataset, divided into bags for MIL learning.

    See `Scikit-learn documentation
    <http://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_hastie_10_2.html>`_
    for more details on the Hastie et al. 2009, Example 10.2 dataset generating method.

    """

    def __init__(self, random_state=1, n=6):

        self.data, self.labels = make_hastie_10_2(random_state=random_state)
        i = np.argsort(self.labels)
        self.labels = self.labels[i]
        self.data = self.data[i, :]

        self.bag_labels = []
        for i, k in enumerate(six.moves.range(0, len(self.labels), n)):
            self.labels[k:k+n] *= i + 1
            self.bag_labels.append(np.sign(np.max(self.labels[k:k+n])))
        self.labels = np.array(self.labels, 'int')
        self.bag_labels = np.array(self.bag_labels, 'int')
        self.bag_partitioning = np.cumsum(np.bincount(np.abs(self.labels))[1:])[:-1]
