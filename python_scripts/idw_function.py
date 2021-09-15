# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:14:37 2021

@author: pearsonra
"""

import multiprocessing
from multiprocessing import shared_memory
from numba import jit
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


class InvDistTreeFunc:
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

    def __init__(self, xy: numpy.ndarray, z: numpy.ndarray, leafsize: int = 10):
        assert len(xy) == len(z), f"len(X) of {len(xy)} != len(z) {len(z)}"
        self.tree = scipy.spatial.KDTree(xy, leafsize=leafsize)  # build the tree
        assert type(z) is numpy.ndarray and z.ndim == 1, f"Expecting z to be a 1D numpy array instead got {z}"
        self.z = z

    def __call__(self, grid_xy: numpy.ndarray, search_radius, power=1, smoothing=0, eps=0):

        assert type(grid_xy) is numpy.ndarray and grid_xy.ndim == 2, "Expect grid_xy to be a 2D numpy array " \
            f" instead got {grid_xy}"

        tree_index_list = self.tree.query_ball_point(grid_xy, r=search_radius, eps=eps)  # , eps=0.2)
        interpolant = numpy.zeros(len(grid_xy))

        for i, (near_indicies, point) in enumerate(zip(tree_index_list, grid_xy)):

            if len(near_indicies) == 0:  # Set NaN if no values in search region
                interpolant[i] = numpy.nan
            else:
                distance_vectors = point - self.tree.data[near_indicies]
                smoothed_distances = numpy.sqrt(((distance_vectors**2).sum(axis=1)+smoothing**2))
                if smoothed_distances.min() == 0:  # incase the of an exact match
                    interpolant[i] = self.z[self.tree.query(grid_xy[i], k=1)[1]]
                else:
                    interpolant[i] = (self.z[near_indicies] / (smoothed_distances**power)).sum(axis=0) \
                        / (1 / (smoothed_distances**power)).sum(axis=0)

        return interpolant


@jit
def idw_function(xy: numpy.ndarray, z: numpy.ndarray, grid_xy: numpy.ndarray, search_radius: float,
                 leafsize: int = 10, power: int = 1, smoothing: float = 0, eps: float = 0):
    assert len(xy) == len(z), f"len(X) of {len(xy)} != len(z) {len(z)}"
    assert type(z) is numpy.ndarray and z.ndim == 1, f"Expecting z to be a 1D numpy array instead got {z}"
    assert type(grid_xy) is numpy.ndarray and grid_xy.ndim == 2, "Expect grid_xy to be a 2D numpy array " \
        f" instead got {grid_xy}"

    tree = scipy.spatial.KDTree(xy, leafsize=leafsize)  # build the tree
    tree_index_list = tree.query_ball_point(grid_xy, r=search_radius, eps=eps)  # , eps=0.2)
    interpolant = numpy.zeros(len(grid_xy))

    for i, (near_indicies, point) in enumerate(zip(tree_index_list, grid_xy)):

        if len(near_indicies) == 0:  # Set NaN if no values in search region
            interpolant[i] = numpy.nan
        else:
            distance_vectors = point - tree.data[near_indicies]
            smoothed_distances = numpy.sqrt(((distance_vectors**2).sum(axis=1)+smoothing**2))
            if smoothed_distances.min() == 0:  # incase the of an exact match
                interpolant[i] = z[tree.query(grid_xy[i], k=1)[1]]
            else:
                interpolant[i] = (z[near_indicies] / (smoothed_distances**power)).sum(axis=0) \
                    / (1 / (smoothed_distances**power)).sum(axis=0)

    return interpolant


def setup_pdal():

    h_crs = 2193
    v_crs = 7839
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
    print(f"Difference org to new func:{interpolated_z - interpolated_z_org}")

    # print(f"interpolate a single point: {idw_tree(known[0], nnear=Nnear, eps=eps)})


def main():
    measured_z, measured_xy, grid_xy, grid_z = setup()
    idw_interpolate(measured_z, measured_xy, grid_xy, grid_z)

    all_points = setup_pdal()
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
    num_points_in_radius = 8  # 8 2d, 11 3d => 5 % chance one-sided -- Wendel, mathoverflow.com
    eps = .1  # approximate nearest, dist <= (1 + eps) * true nearest
    power = 1

    start_time = time.time()
    idw_tree_num_points = InvDistTreeFuncNumPoints(xy_in, z_in, leafsize=leafsize)
    interpolated_z_num_points = idw_tree_num_points(grid_xy, num_points_in_radius=num_points_in_radius, eps=eps,
                                                    power=power)
    grid_z_num_points = interpolated_z_num_points.reshape(grid_x.shape)
    print(f"Executing nearest points IDW takes {time.time()-start_time}")

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

    fig, axs = matplotlib.pyplot.subplots(1, 3, figsize=(30, 10))
    im_0 = axs[0].imshow(grid_z_num_points)
    axs[0].set_title("100m and points in sqrt(2)*100")
    matplotlib.pyplot.colorbar(im_0, ax=axs[0])

    im_1 = axs[1].imshow(grid_z)
    axs[1].set_title("100m and the 8 nearest points")
    matplotlib.pyplot.colorbar(im_1, ax=axs[1])

    im_2 = axs[2].imshow(grid_z_approximate)
    axs[2].set_title("100m and the 8 nearest points")
    matplotlib.pyplot.colorbar(im_2, ax=axs[2])
    matplotlib.pyplot.show()

    # Try use GDAL
    start_time = time.time()
    h_crs = 2193
    v_crs = 7839
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
    idw_gdal.plot()


    '''# Try use numba
    start_time = time.time()
    interpolated_z_function = idw_function(xy_in, z_in, grid_xy, search_radius=numpy.sqrt(2)*res,
                                           leafsize=leafsize, power=power, eps=eps)
    grid_z_function = interpolated_z_function.reshape(grid_x.shape)
    print(f"Executing all points approximate IDW takes {time.time()-start_time}")

    fig, ax = matplotlib.pyplot.subplots(1, 1, figsize=(30, 21.5))
    im_0 = ax.imshow(grid_z_function)
    ax.set_title("10m and numba")
    matplotlib.pyplot.colorbar(im_0, ax=ax)

    matplotlib.pyplot.show()'''


if __name__ == "__main__":
    main()
