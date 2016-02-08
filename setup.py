# -*- coding: utf-8 -*-
"""
:mod:`setup.py`
===============

.. module:: setup
   :platform: Unix, Windows
   :synopsis: The Python Packaging setup file for MilBoost.

.. moduleauthor:: hbldh <henrik.blidh@nedomkull.com>

Created on 2015-11-05

"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from setuptools import setup, find_packages, Extension

import numpy
import skboost

basedir = os.path.dirname(os.path.abspath(__file__))

with open('README.md') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='skboost',
    version=skboost.__version__,
    author=skboost.author,
    author_email=skboost.author_email,
    maintainer=skboost.maintainer,
    maintainer_email=skboost.maintainer_email,
    url=skboost.url,
    download_url=skboost.download_url,
    description=skboost.description,
    long_description=LONG_DESCRIPTION,
    license=skboost.license,
    platforms=skboost.platforms,
    keywords=skboost.keywords,
    classifiers=skboost.classifiers,
    packages=find_packages(exclude=['tests', 'scripts']),
    package_data={
        'skboost.stumps.ext': ['src/*', ],
        'skboost.datasets.musk': ['clean*.*', ],
    },
    install_requires=[
        'numpy>=1.6.1',
        'scipy>=0.9',
        'scikit-learn>=0.16.0',
        'six>=1.10.0',
        'futures>=3.0.3',
        'psutil>=3.4.2'
    ],
    dependency_links=[],
    ext_package='skboost.stumps.ext',
    ext_modules=[
        Extension('classifiers',
                  sources=['skboost/stumps/ext/src/classifiers.c'],
                  include_dirs=[numpy.get_include(),
                                'skboost/stumps/ext/src/'])
    ],
    entry_points={}
)

