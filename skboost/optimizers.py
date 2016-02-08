#!/usr/bin/env python
# encoding: utf-8
"""
optimizers.py
"""

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np

import logging
import sys


def perform_search(f, bounds, desired_value=0.0, x_tol=1e-5):
    """Performs a binary search.

    N.B. this binary search can only handle monotonically increasing or
    decreasing functions.

    """

    logger = logging.getLogger('Golden Section')
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    logger.debug("Binary Search initiated.")
    l_point = bounds[0]
    r_point = bounds[1]
    l_point_val = f(l_point)
    r_point_val = f(r_point)
    logger.debug("Start interval = [{0} {1}], Values = [{2} {3}].".format( \
        l_point, r_point, l_point_val, r_point_val))
    logger.debug("Searching for x s.t. f(x) = {0}.".format(desired_value))
    # If the function values indicate that the function is monotonically
    # increasing, then reverse the comparison fcn and perform the search
    # for that situation instead.
    if l_point_val < desired_value:
        comp_fcn = np.less
        logger.debug("Assuming increasing evaluation function. "
                     "Using {0} for comparisons".format(str(comp_fcn)))
    else:
        comp_fcn = np.greater
        logger.debug("Assuming decreasing evaluation function. "
                     "Using {0} for comparisons".format(str(comp_fcn)))

    while np.abs(r_point - l_point) > x_tol:
        new_point = (r_point + l_point) / 2
        new_point_val = f(new_point)
        if comp_fcn(new_point_val, desired_value):
            l_point = new_point
            l_point_val = new_point_val
        else:
            r_point = new_point
            r_point_val = new_point_val
        logger.debug("New interval = [{0} {1}], {2}{3} {4}].".format(
            l_point, r_point, "Values = [", l_point_val, r_point_val))
        logger.debug("Interval span = {0}, Stop criterion = {1}.".format(
            np.abs(r_point - l_point), x_tol))

    best_point = np.mean([l_point, r_point])
    logger.debug("Found x = {0} as best point, f(x) = {1}.".format(
        best_point, f(best_point)))

    return best_point


def golden_section_search(f, bounds, x_tol=1e-5):
    """Performs a golden section search."""

    logger = logging.getLogger('Golden Section')
    logger.addHandler(logging.StreamHandler(stream=sys.stdout))

    logger.debug("Golden Section Search initiated.")
    gm = (1 + np.sqrt(5)) / 2
    points = np.array([bounds[0],
                       bounds[1] - (bounds[1] - bounds[0]) / gm,
                       bounds[0] + bounds[1] / gm,
                       bounds[1]], 'float')
    vals = np.zeros(4, 'float')
    visited_points = []
    visited_points_vals = []

    cmp_fcn = np.less_equal
    end_eval_fcn = np.argmin
    new_val = np.inf

    logger.debug("Start interval = {0}.".format(points))
    while np.abs(points[3] - points[0]) > x_tol:
        # Evaluate points.
        for i, pnt in enumerate(points):
            if pnt in visited_points:
                continue
            vals[i] = f(pnt)
            visited_points.append(pnt)
            visited_points_vals.append(vals[i])

        if cmp_fcn(vals[1], vals[2]):
            points = np.array([points[0], points[2] - (points[2] - points[0]) / gm, points[1],
                               points[2]], 'float')
            vals = np.array([vals[0], new_val, vals[1], vals[2]])
        else:
            points = np.array([points[1], points[2], points[1] + ((points[3] - points[1]) / gm),
                               points[3]], 'float')
            vals = np.array([vals[1], vals[2], new_val, vals[3]])
        logger.debug("New interval = {0}, Values = {1}.".format(
            points, vals))
        logger.debug("Interval span = {0}, Stop criterion = {1}.".format(
            np.abs(points[3] - points[0]), x_tol))

    best_ind = end_eval_fcn(vals)
    best_point = points[best_ind]
    logger.debug("Found x = {0} as best point, f(x) = {1}.".format(
        best_point, vals[best_ind]))

    return best_point
