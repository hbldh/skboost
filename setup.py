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

import re
import os
from codecs import open
from setuptools import setup, find_packages, Extension

import numpy

basedir = os.path.dirname(os.path.abspath(__file__))


def read(f):
    return open(f, encoding='utf-8').read()


with open('skboost/version.py', 'r') as fd:
    version = re.search(
        '^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        fd.read(), re.MULTILINE).group(1)

setup(
    name='skboost',
    version=version,
    author='Henrik Blidh',
    author_email='henrik.blidh@nedomkull.com',
    url='https://github.com/hbldh/skboost',
    description="Boosting Algorithms compatible with scikit-learn",
    long_description=read('README.md'),
    license='MIT',
    keywords=['Machine Learning', 'Boosting'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: MIT',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['tests', 'scripts']),
    package_data={
        'skboost.stumps.ext': ['src/*', ],
        'skboost.datasets.musk': ['clean*.*', ],
    },
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'six',
        'psutil',
        'futures;python_version<"3.4"',
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

