#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
:mod:`musk`
===========

.. module:: musk
    :platform: Unix, Windows
    :synopsis:

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-06, 14:11

"""
from pathlib import Path
import numpy as np

__all__ = ["MUSK1", "MUSK2"]


class _MUSK(object):
    def __init__(self, id_):
        self.name = None
        data, self._names, self.info = _read_musk_data(id_)
        self.data, self.labels, (self.molecule_names, self.conformation_names) = _parse_musk_data(data)

        unique_bag_ids = np.unique(self.labels)
        self.bag_labels = np.zeros((max(np.abs(unique_bag_ids)) + 1,), "int")
        self.bag_labels[np.abs(unique_bag_ids)] = np.sign(unique_bag_ids)
        self.bag_labels = self.bag_labels[1:]
        self.bag_partitioning = np.cumsum(np.bincount(np.abs(self.labels))[1:])[:-1]

    def __str__(self):
        return "{0} - {1} instances with {2} features. {3} Musks, {4} Non-musks".format(
            self.name,
            self.data.shape[0],
            self.data.shape[1],
            len(np.unique(self.labels[self.labels > 0])),
            len(np.unique(self.labels[self.labels < 0])),
        )

    def __repr__(self):
        return str(self)

    def __unicode__(self):
        return str(self)


class MUSK1(_MUSK):
    def __init__(self):
        super(MUSK1, self).__init__(1)
        self.name = 'MUSK "Clean1" database'


class MUSK2(_MUSK):
    def __init__(self):
        super(MUSK2, self).__init__(2)
        self.name = 'MUSK "Clean2" database'


def _read_musk_data(id_):
    _here = Path(__file__).parent

    with open(_here / f"clean{id_}.data", mode="r") as f:
        data = f.readlines()
    with open(_here / f"clean{id_}.names", mode="r") as f:
        names = f.readlines()
    with open(_here / f"clean{id_}.info", mode="r") as f:
        info = f.read()
    return data, names, info


def _parse_musk_data(data):
    molecule_names = []
    conformation_names = []
    f = []
    labels = []

    bag_id = 0
    for row in data:
        items = row.strip("\n").split(",")
        if items[0] not in molecule_names:
            molecule_names.append(items[0])
            bag_id += 1
        labels.append(int(bag_id * ((float(items[-1]) * 2) - 1)))
        conformation_names.append(items[1])
        f.append(list(map(float, items[2:-1])))

    return np.array(f), np.array(labels), (molecule_names, conformation_names)
