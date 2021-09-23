# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:14:37 2021

@author: pearsonra
"""

#from numba import jit
import dask
import dask.array
import dask.distributed
import sklearn
import sklearn.neighbors
import numpy
import scipy.spatial  # import cKDTree as KDTree
import pathlib
import time
import math
import typing

import pdal
import json
import rioxarray

import matplotlib.pyplot
#import contextlib.closing


class Invdisttreeorg:
    """ inverse-distance-weighted interpolation using KDTree:
    invdisttree = Invdisttree( X, z )  -- data points, values
    interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
        interpolates z from the 3 points nearest each query point q;
        For example, interpol[ a query point q ]
        finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
            (z1/d1 + z2/d2 + z3/d3)
            / (1/d1 + 1/d2 + 1/d3)
            = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3
    
        q may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        p: use 1 / distance**p
        weights: optional multipliers for 1 / distance**p, of the same shape as q
        stat: accumulate wsum, wn for average weights
    
    How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.
    
    p=1, p=2 ?
        p=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so p=2 is inverse-area weighting. For example,
            (z1/area1 + z2/area2 + z3/area3)
            / (1/area1 + 1/area2 + 1/area3)
            = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, p=3 is inverse-volume weighting.
    
    Scaling:
        if different X coordinates measure different things, Euclidean distance
        can be way off.  For example, if X0 is in the range 0 to 1
        but X1 0 to 1000, the X1 distances will swamp X0;
        rescale the data, i.e. make X0.std() ~= X1.std() .
    
    A nice property of IDW is that it's scale-free around query points:
    if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
    the IDW average
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
    is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
    In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
    is exceedingly sensitive to distance and to h.
    """
    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__(self, X, z, leafsize=10, stat=0):
        assert len(X) == len(z), "len(X) %d != len(z) %d" % (len(X), len(z))
        self.tree = scipy.spatial.cKDTree(X, leafsize=leafsize)  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None

    def __call__(self, q, nnear=6, eps=0, p=1, weights=None):
        # nnear nearest neighbours of each query point --
        q = numpy.asarray(q)
        qdim = q.ndim
        if qdim == 1:
            q = numpy.array([q])
        if self.wsum is None:
            self.wsum = numpy.zeros(nnear)

        self.distances, self.ix = self.tree.query(q, k=nnear, eps=eps)
        interpol = numpy.zeros((len(self.distances),) + numpy.shape(self.z[0]))
        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if nnear == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**p
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= numpy.sum(w)
                wz = numpy.dot(w, self.z[ix])
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1 else interpol[0]


class InvDistTreeFuncAll:
    """ inverse-distance-weighted interpolation using KDTree:
    idw_tree = InvDistTreeFunc(xy, z)  -- data points, values
    interpol = idw_tree(q, num_points_in_radius=3, eps=0, power=1, weights=None, stat=0)
        interpolates z from the num_points_in_radius=3 points nearest each query point q;
        For example, interpol[ a query point q ]
        finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
            (z1/d1 + z2/d2 + z3/d3)
            / (1/d1 + 1/d2 + 1/d3)
            = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

        grid_xy may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        power: use 1 / distance**power
        weights: optional multipliers for 1 / distance**power, of the same shape as q
        stat: accumulate wsum, wn for average weights

    num_points_in_radius - How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.

    power=1, power=2 ?
    power=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so power=2 is inverse-area weighting. For example,
            (z1/area1 + z2/area2 + z3/area3) / (1/area1 + 1/area2 + 1/area3)
            = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, power=3 is inverse-volume weighting.

    Scaling:
        if different xy coordinates measure different things, Euclidean distance
        can be way off.  For example, if x is in the range 0 to 1
        but y is 0 to 1000, the y distances will swamp the x;
        rescale the data, i.e. make x.std() ~= y.std() .

    A nice property of IDW is that it's scale-free around query points:
    if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
    the IDW average: (z1/d1 + z2/d2 + z3/d3) / (1/d1 + 1/d2 + 1/d3)
    is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
    In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
    is exceedingly sensitive to distance and to h.
    """
    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__(self, xy: numpy.ndarray, z: numpy.ndarray, leafsize: int = 10, stat=0):
        assert len(xy) == len(z), f"len(X) of {len(xy)} != len(z) {len(z)}"
        self.tree = scipy.spatial.cKDTree(xy, leafsize=leafsize)  # build the tree
        self.z = z
        self.stat = stat
        self.wn = 0
        self.wsum = None

    def __call__(self, grid_xy: numpy.ndarray, num_points_in_radius=6, eps=0, power=1, weights=None):
        # nnear nearest neighbours of each query point --
        grid_xy = numpy.asarray(grid_xy)
        qdim = grid_xy.ndim
        if qdim == 1:
            grid_xy = numpy.array([grid_xy])
        if self.wsum is None:
            self.wsum = numpy.zeros(num_points_in_radius)

        self.distances, self.ix = self.tree.query(grid_xy, k=num_points_in_radius, eps=eps)
        interpol = numpy.zeros((len(self.distances),) + numpy.shape(self.z[0]))
        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if num_points_in_radius == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z s by 1/dist --
                w = 1 / dist**power
                if weights is not None:
                    w *= weights[ix]  # >= 0
                w /= numpy.sum(w)
                wz = numpy.dot(w, self.z[ix])
                if self.stat:
                    self.wn += 1
                    self.wsum += w
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol if qdim > 1 else interpol[0]


class InvDistTreeFuncNumPoints:
    """ inverse-distance-weighted interpolation using KDTree:
    idw_tree = InvDistTreeFunc(xy, z)  -- data points, values
    interpol = idw_tree(q, num_points_in_radius=3, eps=0, power=1, weights=None, stat=0)
        interpolates z from the num_points_in_radius=3 points nearest each query point q;
        For example, interpol[ a query point q ]
        finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
            (z1/d1 + z2/d2 + z3/d3)
            / (1/d1 + 1/d2 + 1/d3)
            = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

        grid_xy may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        power: use 1 / distance**power
        weights: optional multipliers for 1 / distance**power, of the same shape as q

    num_points_in_radius - How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.

    power=1, power=2 ?
    power=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so power=2 is inverse-area weighting. For example,
            (z1/area1 + z2/area2 + z3/area3) / (1/area1 + 1/area2 + 1/area3)
            = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, power=3 is inverse-volume weighting.

    Scaling:
        if different xy coordinates measure different things, Euclidean distance
        can be way off.  For example, if x is in the range 0 to 1
        but y is 0 to 1000, the y distances will swamp the x;
        rescale the data, i.e. make x.std() ~= y.std() .

    A nice property of IDW is that it's scale-free around query points:
    if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
    the IDW average: (z1/d1 + z2/d2 + z3/d3) / (1/d1 + 1/d2 + 1/d3)
    is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
    In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
    is exceedingly sensitive to distance and to h.
    """
    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__(self, xy: numpy.ndarray, z: numpy.ndarray, leafsize: int = 10, stat=0):
        assert len(xy) == len(z), f"len(X) of {len(xy)} != len(z) {len(z)}"
        self.tree = scipy.spatial.KDTree(xy, leafsize=leafsize)  # build the tree
        assert type(z) is numpy.ndarray and z.ndim == 1, f"Expecting z to be a 1D numpy array instead got {z}"
        self.z = z

    def __call__(self, grid_xy: numpy.ndarray, num_points_in_radius=6, eps=0, power=1, smoothing=0):

        assert type(grid_xy) is numpy.ndarray and grid_xy.ndim == 2, f"Expecting a grid_xy to be a 2D numpy array instead got {grid_xy}"

        self.distances, self.ix = self.tree.query(grid_xy, k=num_points_in_radius, eps=eps)
        interpol = numpy.zeros(len(grid_xy))
        jinterpol = 0
        for dist, ix in zip(self.distances, self.ix):
            if num_points_in_radius == 1:
                wz = self.z[ix]
            elif dist[0] < 1e-10:
                wz = self.z[ix[0]]
            else:  # weight z's by 1/dist --
                w = 1 / dist**power
                w /= numpy.sum(w)
                wz = numpy.dot(w, self.z[ix])
            interpol[jinterpol] = wz
            jinterpol += 1
        return interpol


class InvDistTreeFuncSciPy:
    """ inverse-distance-weighted interpolation using KDTree:
    idw_tree = InvDistTreeFunc(xy, z)  -- data points, values
    interpol = idw_tree(q, num_points_in_radius=3, eps=0, power=1, weights=None, stat=0) interpolates z from the
    num_points_in_radius=3 points nearest each query point q;
        For example, interpol[ a query point q ] finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
            (z1/d1 + z2/d2 + z3/d3) / (1/d1 + 1/d2 + 1/d3)  = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

        grid_xy: may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        power: use 1 / distance**power

    num_points_in_radius - How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results -- |interpol 6 - interpol 8| etc.,
    or |f - interpol*| if you have f(q).  I find that runtimes don't increase much at all with nnear -- ymmv.

    power=1, power=2 ?
    power=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2, so power=2 is inverse-area weighting.
        For example, (z1/area1 + z2/area2 + z3/area3) / (1/area1 + 1/area2 + 1/area3) = .74 z1 + .18 z2 + .08 z3
        for distances 1 2 3. Similarly, in 3d, power=3 is inverse-volume weighting.

    Scaling:
        if different xy coordinates measure different things, Euclidean distance can be way off.
        For example, if x is in the range 0 to 1 but y is 0 to 1000, the y distances will swamp the x;
        rescale the data, i.e. make x.std() ~= y.std() .

    A nice property of IDW is that it's scale-free around query points: if I have values z1 z2 z3 from 3 points
    at distances d1 d2 d3, the IDW average: (z1/d1 + z2/d2 + z3/d3) / (1/d1 + 1/d2 + 1/d3) is the same for
    distances 1 2 3, or 10 20 30 -- only the ratios matter. In contrast, the commonly-used Gaussian kernel
    exp( - (distance/h)**2 ) is exceedingly sensitive to distance and to h.
    """
    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__(self, xy_in: numpy.ndarray, z_in: numpy.ndarray, leafsize: int = 10):
        assert len(xy_in) == len(z_in), f"len(X) of {len(xy_in)} != len(z_in) {len(z_in)}"
        self.tree = scipy.spatial.KDTree(xy_in, leafsize=leafsize)  # build the tree
        assert type(z_in) is numpy.ndarray and z_in.ndim == 1, f"Expecting z_in to be a 1D numpy array, but got {z_in}"
        self.z_in = z_in

    def __call__(self, xy_out: numpy.ndarray, search_radius, power=1, smoothing=0, eps=0):

        assert type(xy_out) is numpy.ndarray and xy_out.ndim == 2, "Expect xy_out to be a 2D numpy array " \
            f" instead got {xy_out}"

        tree_index_list = self.tree.query_ball_point(xy_out, r=search_radius, eps=eps)  # , eps=0.2)
        interpolant = numpy.zeros(len(xy_out))

        for i, (near_indicies, point) in enumerate(zip(tree_index_list, xy_out)):

            if len(near_indicies) == 0:  # Set NaN if no values in search region
                interpolant[i] = numpy.nan
            else:
                distance_vectors = point - self.tree.data[near_indicies]
                smoothed_distances = numpy.sqrt(((distance_vectors**2).sum(axis=1)+smoothing**2))
                if smoothed_distances.min() == 0:  # incase the of an exact match
                    interpolant[i] = self.z_in[self.tree.query(point, k=1)[1]]
                else:
                    interpolant[i] = (self.z_in[near_indicies] / (smoothed_distances**power)).sum(axis=0) \
                        / (1 / (smoothed_distances**power)).sum(axis=0)

        return interpolant


class InvDistTreeFuncLearn:
    """ inverse-distance-weighted interpolation using KDTree:
    idw_tree = InvDistTreeFunc(xy, z)  -- data points, values
    interpol = idw_tree(q, num_points_in_radius=3, eps=0, power=1, weights=None, stat=0) interpolates z from the
    num_points_in_radius=3 points nearest each query point q;
        For example, interpol[ a query point q ] finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
            (z1/d1 + z2/d2 + z3/d3) / (1/d1 + 1/d2 + 1/d3)  = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

        grid_xy: may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        power: use 1 / distance**power

    num_points_in_radius - How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results -- |interpol 6 - interpol 8| etc.,
    or |f - interpol*| if you have f(q).  I find that runtimes don't increase much at all with nnear -- ymmv.

    power=1, power=2 ?
    power=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2, so power=2 is inverse-area weighting.
        For example, (z1/area1 + z2/area2 + z3/area3) / (1/area1 + 1/area2 + 1/area3) = .74 z1 + .18 z2 + .08 z3
        for distances 1 2 3. Similarly, in 3d, power=3 is inverse-volume weighting.

    Scaling:
        if different xy coordinates measure different things, Euclidean distance can be way off.
        For example, if x is in the range 0 to 1 but y is 0 to 1000, the y distances will swamp the x;
        rescale the data, i.e. make x.std() ~= y.std() .

    A nice property of IDW is that it's scale-free around query points: if I have values z1 z2 z3 from 3 points
    at distances d1 d2 d3, the IDW average: (z1/d1 + z2/d2 + z3/d3) / (1/d1 + 1/d2 + 1/d3) is the same for
    distances 1 2 3, or 10 20 30 -- only the ratios matter. In contrast, the commonly-used Gaussian kernel
    exp( - (distance/h)**2 ) is exceedingly sensitive to distance and to h.
    """
    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__(self, xy_in: numpy.ndarray, z_in: numpy.ndarray, leafsize: int = 10):
        assert len(xy_in) == len(z_in), f"len(X) of {len(xy_in)} != len(z_in) {len(z_in)}"
        self.tree = sklearn.neighbors.KDTree(xy_in, leaf_size=leafsize)  # build the tree
        self.xy_in = xy_in
        assert type(z_in) is numpy.ndarray and z_in.ndim == 1, f"Expecting z_in to be a 1D numpy array, but got {z_in}"
        self.z_in = z_in

    def __call__(self, xy_out: numpy.ndarray, search_radius, power=1, smoothing=0, eps=0):

        assert type(xy_out) is numpy.ndarray and xy_out.ndim == 2, "Expect xy_out to be a 2D numpy array " \
            f" instead got {xy_out}"

        tree_index_list = self.tree.query_radius(xy_out, r=search_radius, return_distance=False, count_only=False)
        interpolant = numpy.zeros(len(xy_out))

        for i, (near_indicies, point) in enumerate(zip(tree_index_list, xy_out)):

            if len(near_indicies) == 0:  # Set NaN if no values in search region
                interpolant[i] = numpy.nan
            else:
                distance_vectors = point - self.xy_in[near_indicies]
                smoothed_distances = numpy.sqrt(((distance_vectors**2).sum(axis=1)+smoothing**2))
                if smoothed_distances.min() == 0:  # incase the of an exact match
                    interpolant[i] = self.z_in[self.tree.query([point], k=1, return_distance=False)[0][0]]
                else:
                    interpolant[i] = (self.z_in[near_indicies] / (smoothed_distances**power)).sum(axis=0) \
                        / (1 / (smoothed_distances**power)).sum(axis=0)

        return interpolant


def idw_function_scipy(xy_in: numpy.ndarray, z_in: numpy.ndarray, xy_out: numpy.ndarray, search_radius: float,
                 leafsize: int = 10, power: int = 1, smoothing: float = 0, eps: float = 0):
    assert len(xy_in) == len(z_in), f"len(X) of {len(xy_in)} != len(z_in) {len(z_in)}"
    assert type(z_in) is numpy.ndarray and z_in.ndim == 1, f"Expecting z_in to be a 1D numpy array instead got {z_in}"
    assert type(xy_out) is numpy.ndarray and xy_out.ndim == 2, "Expect xy_out to be a 2D numpy array " \
        f" instead got {xy_out}"

    tree = scipy.spatial.KDTree(xy_in, leafsize=leafsize)  # build the tree
    tree_index_list = tree.query_ball_point(xy_out, r=search_radius, eps=eps)  # , eps=0.2)
    z_out = numpy.zeros(len(xy_out))

    for i, (near_indicies, point) in enumerate(zip(tree_index_list, xy_out)):

        if len(near_indicies) == 0:  # Set NaN if no values in search region
            z_out[i] = numpy.nan
        else:
            distance_vectors = point - tree.data[near_indicies]
            smoothed_distances = numpy.sqrt(((distance_vectors**2).sum(axis=1)+smoothing**2))
            if smoothed_distances.min() == 0:  # incase the of an exact match
                z_out[i] = z_in[tree.query(point, k=1)[1]]
            else:
                z_out[i] = (z_in[near_indicies] / (smoothed_distances**power)).sum(axis=0) \
                    / (1 / (smoothed_distances**power)).sum(axis=0)

    return z_out


def idw_function_learn(xy_in: numpy.ndarray, z_in: numpy.ndarray, xy_out: numpy.ndarray, search_radius: float,
                 leafsize: int = 10, power: int = 1, smoothing: float = 0, eps: float = 0):
    assert len(xy_in) == len(z_in), f"len(X) of {len(xy_in)} != len(z_in) {len(z_in)}"
    assert type(z_in) is numpy.ndarray and z_in.ndim == 1, f"Expecting z_in to be a 1D numpy array instead got {z_in}"
    assert type(xy_out) is numpy.ndarray and xy_out.ndim == 2, "Expect xy_out to be a 2D numpy array " \
        f" instead got {xy_out}"

    tree = sklearn.neighbors.KDTree(xy_in, leaf_size=leafsize)  # build the tree
    tree_index_array = tree.query_radius(xy_out, r=search_radius, return_distance=False, count_only=False)
    z_out = numpy.zeros(len(xy_out))

    for i, (near_indicies, point) in enumerate(zip(tree_index_array, xy_out)):

        if len(near_indicies) == 0:  # Set NaN if no values in search region
            z_out[i] = numpy.nan
        else:
            distance_vectors = point - xy_in[near_indicies]
            smoothed_distances = numpy.sqrt(((distance_vectors**2).sum(axis=1)+smoothing**2))
            if smoothed_distances.min() == 0:  # incase the of an exact match
                z_out[i] = z_in[tree.query([point], k=1, return_distance=False)[0][0]]
            else:
                z_out[i] = (z_in[near_indicies] / (smoothed_distances**power)).sum(axis=0) \
                    / (1 / (smoothed_distances**power)).sum(axis=0)

    return z_out


def setup_pdal(h_crs, v_crs):

    LAS_GROUND = 2
    tile_name = pathlib.Path(
        r"C:\Users\pearsonra\Documents\data\test_parallel\cache\Wellington_2013\ot_CL1_WLG_2013_1km_083039.laz")

    pdal_pipeline_instructions = [{"type":  "readers.las", "filename": str(tile_name)},
                                  {"type": "filters.reprojection", "in_srs": f"EPSG:{h_crs}+{v_crs}",
                                  "out_srs": f"EPSG:{h_crs}+{v_crs}"}]

    # Load in LiDAR and perform operations
    pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions))
    pdal_pipeline.execute()

    # Load LiDAR points from pipeline
    all_points = pdal_pipeline.arrays[0]
    ground_points = all_points[all_points['Classification'] == LAS_GROUND]
    return all_points


def setup():

    N = 10000
    Ndim = 2
    Nask = N  # N Nask 1e5: 24 sec 2d, 27 sec 3d on mac g4 ppc
    cycle = .25
    seed = 1

    # exec "\n".join( sys.argv[1:] )  # python this.py N= ...
    numpy.random.seed(seed)
    numpy.set_printoptions(3, threshold=100, suppress=True)  # .3f

    print(f"Paramters:  N {N}  Ndim {Ndim}  Nask {Nask}")

    def terrain(x):
        """ ~ rolling hills """
        return numpy.sin((2 * numpy.pi / cycle) * numpy.mean(x, axis=-1))

    known = numpy.random.uniform(size=(N, Ndim)) ** .5  # 1/(p+1): density x^p
    z = terrain(known)
    ask = numpy.random.uniform(size=(Nask, Ndim))
    ask_terrain = terrain(ask)

    return z, known, ask, ask_terrain


def idw_interpolate(measured_z, measured_xy, grid_xy, grid_z):

    # note PDAL - window=0, radius = sqrt(2)*resolution, power = 2
    # window = cells to do backup interpolation in, radius used to say points are in pixel
    # GDAL implementation - all points used in the search radius

    leafsize = 10
    num_points_in_radius = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    power = 1  # weights ~ 1 / distance**p

    print(f"Parameters:  num_points_in_radius {num_points_in_radius}  leafsize {leafsize}  eps {eps}  power {power}")

    idw_tree_org = Invdisttreeorg(measured_xy, measured_z, leafsize=leafsize, stat=1)
    interpolated_z_org = idw_tree_org(grid_xy, nnear=num_points_in_radius, eps=eps, p=power)

    idw_tree = InvDistTreeFuncNumPoints(measured_xy, measured_z, leafsize=leafsize)
    interpolated_z = idw_tree(grid_xy, num_points_in_radius=num_points_in_radius, eps=eps, power=power)

    print(f"average distances to nearest points: {numpy.mean(idw_tree.distances, axis=0)}")
    print(f"average weights: {idw_tree_org.wsum / idw_tree_org.wn}")  # see Wikipedia Zipf's law
    err = numpy.abs(grid_z - interpolated_z)
    print(f"average |terrain() - interpolated|: {numpy.mean(err)}")
    print(f"Difference org to new func: {(interpolated_z - interpolated_z_org).max()}")

    # print(f"interpolate a single point: {idw_tree(known[0], nnear=Nnear, eps=eps)})


def main():
    h_crs = 2193
    v_crs = 7839

    measured_z, measured_xy, grid_xy, grid_z = setup()
    idw_interpolate(measured_z, measured_xy, grid_xy, grid_z)

    all_points = setup_pdal(h_crs, v_crs)
    xy_in = numpy.empty((len(all_points), 2))
    xy_in[:, 0] = all_points['X']
    xy_in[:, 1] = all_points['Y']
    z_in = all_points['Z']

    res = 10
    grid_x, grid_y = numpy.meshgrid(numpy.arange(xy_in[:, 0].min(), xy_in[:, 0].max(), res),
                                    numpy.arange(xy_in[:, 1].min(), xy_in[:, 1].max(), res))
    grid_xy = numpy.concatenate((grid_x.flatten().reshape((1, -1)), grid_y.flatten().reshape((1, -1))),
                                axis=0).transpose()

    leafsize = 10
    num_points_in_radius = 16  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    power = 1

    start_time = time.time()
    idw_tree_num_points = InvDistTreeFuncNumPoints(xy_in, z_in, leafsize=leafsize)
    interpolated_z_num_points = idw_tree_num_points(grid_xy, num_points_in_radius=num_points_in_radius, eps=eps,
                                                    power=power)
    grid_z_num_points = interpolated_z_num_points.reshape(grid_x.shape)
    print(f"Executing nearest {num_points_in_radius} points IDW takes {time.time()-start_time}")

    start_time = time.time()
    idw_tree = InvDistTreeFunc(xy_in, z_in, leafsize=leafsize)
    interpolated_z = idw_tree(grid_xy, search_radius=numpy.sqrt(2)*res, power=power)
    grid_z = interpolated_z.reshape(grid_x.shape)
    print(f"Executing all points IDW takes {time.time()-start_time}")

    start_time = time.time()
    idw_tree = InvDistTreeFunc(xy_in, z_in, leafsize=leafsize)
    interpolated_z_approximate = idw_tree(grid_xy, search_radius=numpy.sqrt(2)*res, power=power, eps=eps)
    grid_z_approximate = interpolated_z_approximate.reshape(grid_x.shape)
    print(f"Executing all points approximate IDW takes {time.time()-start_time}")
    print(f"Max difference for approximate is {((interpolated_z_approximate-interpolated_z)[numpy.isnan(interpolated_z_approximate-interpolated_z)==False]).max()}")

    # Try use GDAL
    start_time = time.time()
    window_size = 0
    idw_power = 2
    radius = numpy.sqrt(2)*res
    start_time = time.time()
    pdal_pipeline_instructions = [
        {"type":  "writers.gdal", "resolution": res,
         "gdalopts": f"a_srs=EPSG:{h_crs}+{v_crs}", "output_type": ["idw"],
         "filename": r"C:\Users\pearsonra\Documents\data\test_parallel\cache\Wellington_2013\test.tif",
         "window_size": window_size, "power": idw_power, "radius": radius,
         "origin_x": grid_x.min(), "origin_y": grid_y.min(),
         "width": grid_x.shape[0], "height": grid_x.shape[1]}
    ]
    pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [all_points])
    pdal_pipeline.execute()
    with rioxarray.rioxarray.open_rasterio(r"C:\Users\pearsonra\Documents\data\test_parallel\cache\Wellington_2013\test.tif", masked=True) as idw_gdal:
        idw_gdal.load()
    idw_gdal = idw_gdal.copy(deep=True)  # Deep copy is required to ensure the opened file is properly unlocked
    idw_gdal.rio.set_crs(h_crs)
    print(f"GDAL IDW takes {time.time()-start_time}")

    vmin = 60
    vmax = 220

    fig, axs = matplotlib.pyplot.subplots(2, 2, figsize=(30, 30))

    im_0 = axs[0, 0].imshow(grid_z, vmin=vmin, vmax=vmax)
    axs[0, 0].set_title("100m and points in sqrt(2)*100")
    matplotlib.pyplot.colorbar(im_0, ax=axs[0, 0])

    im_1 = axs[0, 1].imshow(grid_z_num_points, vmin=vmin, vmax=vmax)
    axs[0, 1].set_title(f"100m and the {num_points_in_radius} nearest points")
    matplotlib.pyplot.colorbar(im_1, ax=axs[0, 1])

    im_2 = axs[1, 0].imshow(grid_z_approximate, vmin=vmin, vmax=vmax)
    axs[1, 0].set_title("100m and points in sqrt(2)*100 approximate")
    matplotlib.pyplot.colorbar(im_2, ax=axs[1, 0])

    idw_gdal.plot(ax=axs[1, 1], vmin=vmin, vmax=vmax)
    axs[1, 1].set_title('PDAL - writers GDAL IDW')

    matplotlib.pyplot.show()

    # Try use dask
    start_time = time.time()
    interpolated_z_function = idw_function(xy_in, z_in, grid_xy, search_radius=numpy.sqrt(2)*res,
                                           leafsize=leafsize, power=power, eps=eps)
    grid_z_function = interpolated_z_function.reshape(grid_x.shape)
    print(f"Executing all points approximate IDW takes {time.time()-start_time}")

    fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(30, 21.5))
    im_0 = ax.imshow(grid_z_function, vmin=vmin, vmax=vmax)
    ax.set_title("10m and numba")
    matplotlib.pyplot.colorbar(im_0, ax=ax)

    matplotlib.pyplot.show()

    print(f"The max between the class and function is {(grid_z_function - grid_z_approximate)[numpy.isnan(grid_z)==False].max().max()}")


def main_dask():

    # parameters
    res = 10
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    power = 2
    radius = numpy.sqrt(2)*res

    client = dask.distributed.Client(processes=False, threads_per_worker=4, n_workers=1, memory_limit='5GB')
    print(client)

    all_points = setup_pdal()
    xy_in = dask.array.empty((len(all_points), 2))
    xy_in[:, 0] = all_points['X']
    xy_in[:, 1] = all_points['Y']
    z_in = all_points['Z']

    grid_x, grid_y = numpy.meshgrid(numpy.arange(xy_in[:, 0].min(), xy_in[:, 0].max(), res),
                                    numpy.arange(xy_in[:, 1].min(), xy_in[:, 1].max(), res))
    grid_xy = dask.array.concatenate((grid_x.flatten().reshape((1, -1)), grid_y.flatten().reshape((1, -1))),
                                axis=0).transpose()

    start_time = time.time()
    idw_tree = InvDistTreeFunc(xy_in, z_in, leafsize=leafsize)
    interpolated_z_approximate = idw_tree(grid_xy, search_radius=radius, power=power, eps=eps)
    grid_z_class = interpolated_z_approximate.reshape(grid_x.shape)
    print(f"Executing all points approximate IDW takes {time.time()-start_time}")

    # Try use dask
    start_time = time.time()
    interpolated_z_function = idw_function(xy_in, z_in, grid_xy, search_radius=numpy.sqrt(2)*res,
                                           leafsize=leafsize, power=power, eps=eps)
    grid_z_function = interpolated_z_function.reshape(grid_x.shape)
    print(f"Executing all points approximate IDW takes {time.time()-start_time}")

    print(f"The max between the class and function is {(grid_z_function - grid_z_class)[numpy.isnan(grid_z_class)==False].max().max()}")


def main_compare():

    # parameters
    h_crs = 2193
    v_crs = 7839
    res = 10
    leafsize = 10
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    power = 2
    window_size = 0
    radius = numpy.sqrt(2)*res

    # read in lidar
    all_points = setup_pdal(h_crs, v_crs)
    xy_in = numpy.empty((len(all_points), 2))
    xy_in[:, 0] = all_points['X']
    xy_in[:, 1] = all_points['Y']
    z_in = all_points['Z']
    
    iterations = 10
    time_gdal=[]
    time_scipy=[]
    time_sklearn=[]

    # setup intial evaluation grid
    grid_x, grid_y = numpy.meshgrid(numpy.arange(xy_in[:, 0].min(), xy_in[:, 0].max(), res),
                                    numpy.arange(xy_in[:, 1].min(), xy_in[:, 1].max(), res))

    # PDAL/GDAL IDW
    start_time = time.time()
    pdal_pipeline_instructions = [
        {"type":  "writers.gdal", "resolution": res,
         "gdalopts": f"a_srs=EPSG:{h_crs}+{v_crs}", "output_type": ["idw"],
         "filename": r"C:\Users\pearsonra\Documents\data\test_parallel\cache\Wellington_2013\test.tif",
         "window_size": window_size, "power": power, "radius": radius,
         "origin_x":  all_points['X'].min(), "origin_y": all_points['Y'].min(),
         "width": grid_x.shape[0], "height": grid_x.shape[1]}]
    pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [all_points])
    pdal_pipeline.execute()
    with rioxarray.rioxarray.open_rasterio(r"C:\Users\pearsonra\Documents\data\test_parallel\cache\Wellington_2013\test.tif", masked=True) as idw_gdal:
        idw_gdal.load()
    idw_gdal = idw_gdal.copy(deep=True)  # Deep copy is required to ensure the opened file is properly unlocked
    idw_gdal.rio.set_crs(h_crs)
    print(f"GDAL IDW takes {time.time()-start_time}")

    # setup evaluation grid
    grid_x, grid_y = numpy.meshgrid(idw_gdal.x, idw_gdal.y)
    grid_xy = numpy.concatenate((grid_x.flatten().reshape((1, -1)), grid_y.flatten().reshape((1, -1))),
                                axis=0).transpose()

    for i in range(iterations):
        # PDAL/GDAL IDW
        start_time = time.time()
        pdal_pipeline_instructions = [
            {"type":  "writers.gdal", "resolution": res,
             "gdalopts": f"a_srs=EPSG:{h_crs}+{v_crs}", "output_type": ["idw"],
             "filename": r"C:\Users\pearsonra\Documents\data\test_parallel\cache\Wellington_2013\test.tif",
             "window_size": window_size, "power": power, "radius": radius,
             "origin_x":  float(idw_gdal.x.min())-res/2, "origin_y": float(idw_gdal.y.min())-res/2,
             "width": len(idw_gdal.x), "height": len(idw_gdal.y)}]
        pdal_pipeline = pdal.Pipeline(json.dumps(pdal_pipeline_instructions), [all_points])
        pdal_pipeline.execute()
        with rioxarray.rioxarray.open_rasterio(r"C:\Users\pearsonra\Documents\data\test_parallel\cache\Wellington_2013\test.tif", masked=True) as idw_gdal:
            idw_gdal.load()
        idw_gdal = idw_gdal.copy(deep=True)  # Deep copy is required to ensure the opened file is properly unlocked
        idw_gdal.rio.set_crs(h_crs)
        end_time = time.time()
        time_gdal.append(end_time-start_time)
        print(f"GDAL IDW takes {end_time-start_time}")

        # Python IDW scipy - class
        start_time = time.time()
        idw_tree = InvDistTreeFuncSciPy(xy_in, z_in, leafsize=leafsize)
        z_out_flat = idw_tree(grid_xy, search_radius=radius, power=power)  #, eps=eps)
        z_out = z_out_flat.reshape(grid_x.shape)
        end_time = time.time()
        time_scipy.append(end_time-start_time)
        print(f"IDW scipy class takes {end_time-start_time}")
        idw_scipy_class = idw_gdal.copy(deep=True)
        idw_scipy_class.data[0] = z_out.reshape(grid_x.shape)
    
        # Python IDW scipy - function
        start_time = time.time()
        z_out_flat = idw_function_scipy(xy_in, z_in, grid_xy, leafsize=leafsize, search_radius=radius, power=power)  #, eps=eps)
        z_out = z_out_flat.reshape(grid_x.shape)
        print(f"IDW scipy function takes {time.time()-start_time}")
        idw_scipy_function = idw_gdal.copy(deep=True)
        idw_scipy_function.data[0] = z_out.reshape(grid_x.shape)
    
        # Python IDW skikit-learn - class
        start_time = time.time()
        idw_tree = InvDistTreeFuncLearn(xy_in, z_in, leafsize=leafsize)
        z_out_flat = idw_tree(grid_xy, search_radius=radius, power=power)  #, eps=eps)
        z_out = z_out_flat.reshape(grid_x.shape)
        end_time = time.time()
        time_sklearn.append(end_time-start_time)
        print(f"IDW sklearn class takes {end_time-start_time}")
        idw_learn_class = idw_gdal.copy(deep=True)
        idw_learn_class.data[0] = z_out.reshape(grid_x.shape)
    
        # Python IDW skikit-learn - function
        start_time = time.time()
        z_out_flat = idw_function_learn(xy_in, z_in, grid_xy, search_radius=radius, power=power, leafsize=leafsize)  #, eps=eps)
        z_out = z_out_flat.reshape(grid_x.shape)
        end_time = time.time()
        print(f"IDW sklearn function takes {time.time()-start_time}")
        idw_learn_function = idw_gdal.copy(deep=True)
        idw_learn_function.data[0] = z_out.reshape(grid_x.shape)
    
    
        idw_diff_gdal = idw_gdal.copy(deep=True)
        idw_diff_gdal.data = idw_gdal.data - idw_scipy_function.data
        
        idw_diff_python = idw_gdal.copy(deep=True)
        idw_diff_python.data = idw_learn_function.data - idw_scipy_function.data
    
        print(f"Difference scipy function to class: {numpy.abs(idw_scipy_class.data - idw_scipy_function.data)[numpy.isnan(idw_scipy_class.data - idw_scipy_function.data)== False].max().max()}")
        print(f"Difference sklearn function to class: {numpy.abs(idw_learn_class.data - idw_learn_function.data)[numpy.isnan(idw_learn_class.data - idw_learn_function.data)== False].max().max()}")
    
        print(f"The max between the PDAL writers.gdal and scipy implementation is {numpy.abs(idw_diff_gdal.data[0])[numpy.isnan(idw_diff_gdal.data[0])==False].max().max()}")
        print(f"The max between the Python KDTree implementations is {numpy.abs(idw_diff_python.data[0])[numpy.isnan(idw_diff_python.data[0])==False].max().max()}")

    print(f"Mean GDAL and std time: {numpy.mean(time_gdal)} and {numpy.std(time_gdal)}")
    print(f"Mean GDAL and std time: {numpy.mean(time_scipy)} and {numpy.std(time_scipy)}")
    print(f"Mean GDAL and std time: {numpy.mean(time_sklearn)} and {numpy.std(time_sklearn)}")


if __name__ == "__main__":
    main_compare()
