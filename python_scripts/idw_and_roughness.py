# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 16:14:37 2021

@author: pearsonra
"""

#from numba import jit
import dask
import numpy
import scipy.spatial  # import cKDTree as KDTree
import pathlib
import time
import math
import typing

import pdal
import json
import rioxarray
import xarray

import matplotlib.pyplot


def idw_function(xy_in: numpy.ndarray, z_in: numpy.ndarray, xy_out: numpy.ndarray, search_radius: float,
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


def roughness_function_mewis(xy_in: numpy.ndarray, z_in: numpy.ndarray, xy_out: numpy.ndarray, search_radius: float,
                             ground_elevations: numpy.ndarray,
                             voxel_heights: numpy.ndarray = numpy.asarray([0, 0.5, 1, 1.5, 2]), leafsize: int = 10,
                             eps: float = 0):
    assert len(xy_in) == len(z_in), f"len(X) of {len(xy_in)} != len(z_in) {len(z_in)}"
    assert type(z_in) is numpy.ndarray and z_in.ndim == 1, f"Expecting z_in to be a 1D numpy array instead got {z_in}"
    assert type(xy_out) is numpy.ndarray and xy_out.ndim == 2, "Expect xy_out to be a 2D numpy array " \
        f" instead got {xy_out}"
    assert voxel_heights[-1] > voxel_heights[0], "The voxel heights must be monotonically increasing"

    tree = scipy.spatial.KDTree(xy_in, leafsize=leafsize)  # build the tree
    tree_index_list = tree.query_ball_point(xy_out, r=search_radius, eps=eps)
    z_out = numpy.empty((len(xy_out), len(voxel_heights) - 1))

    for i, (near_indicies, point) in enumerate(zip(tree_index_list, xy_out)):
     
        # count number of points in each height
        near_indicies = numpy.array(near_indicies)
        near_heights = z_in[near_indicies]
        count_below_height = numpy.array([len(near_indicies[near_heights < threshold_elevation])
                                          for threshold_elevation in voxel_heights + ground_elevations[i]])
        z_out[i] = numpy.log(count_below_height[1:] / count_below_height[:-1]) \
            / (voxel_heights[1:] - voxel_heights[:-1])

    return z_out

def roughness_function_graham(xy_in: numpy.ndarray, z_in: numpy.ndarray, xy_out: numpy.ndarray, search_radius: float,
                              ground_elevations: numpy.ndarray,
                              voxel_heights: numpy.ndarray = numpy.asarray([0, 0.5, 1, 1.5, 2]), leafsize: int = 10,
                              eps: float = 0):
    assert len(xy_in) == len(z_in), f"len(X) of {len(xy_in)} != len(z_in) {len(z_in)}"
    assert type(z_in) is numpy.ndarray and z_in.ndim == 1, f"Expecting z_in to be a 1D numpy array instead got {z_in}"
    assert type(xy_out) is numpy.ndarray and xy_out.ndim == 2, "Expect xy_out to be a 2D numpy array " \
        f" instead got {xy_out}"
    assert voxel_heights[-1] > voxel_heights[0], "The voxel heights must be monotonically increasing"

    tree = scipy.spatial.KDTree(xy_in, leafsize=leafsize)  # build the tree
    tree_index_list = tree.query_ball_point(xy_out, r=search_radius, eps=eps)
    z_out = numpy.empty((len(xy_out), len(voxel_heights) - 1))

    for i, (near_indicies, point) in enumerate(zip(tree_index_list, xy_out)):
     
        # count number of points in each height
        near_indicies = numpy.array(near_indicies)
        near_heights = z_in[near_indicies]
        count_below_height = numpy.array([len(near_indicies[near_heights < threshold_elevation])
                                          for threshold_elevation in voxel_heights + ground_elevations[i]])
        z_out[i] = (count_below_height[1:] - count_below_height[:-1]) / count_below_height[1:] \
            * (voxel_heights[1:] - voxel_heights[:-1])

    return z_out


def setup_pdal(h_crs, v_crs, verbose=True):

    if(verbose):
        print("Loading in Lidar Tile")
    
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
    return all_points, ground_points


def main_compare():
    """ A class to generate IDW and roughness values. """

    verbose = True

    # parameters
    h_crs = 2193
    v_crs = 7839
    res = 10
    leafsize = 10
    eps = 0  # approximate nearest, dist <= (1 + eps) * true nearest
    power = 2
    radius = numpy.sqrt(2)*res
    raster_type = numpy.float64

    # read in lidar
    all_points, gnd_points = setup_pdal(h_crs, v_crs, verbose)

    # ground points
    xy_in = numpy.empty((len(gnd_points), 2))
    xy_in[:, 0] = gnd_points['X']
    xy_in[:, 1] = gnd_points['Y']
    z_in = gnd_points['Z']

    # setup intial evaluation grid
    grid_x, grid_y = numpy.meshgrid(numpy.arange(xy_in[:, 0].min(), xy_in[:, 0].max(), res),
                                    numpy.arange(xy_in[:, 1].min(), xy_in[:, 1].max(), res))
    grid_xy = numpy.concatenate((grid_x.flatten().reshape((1, -1)), grid_y.flatten().reshape((1, -1))),
                                axis=0).transpose()
    dim_x = numpy.arange(xy_in[:, 0].min(), xy_in[:, 0].max(), res)
    dim_y = numpy.arange(xy_in[:, 1].min(), xy_in[:, 1].max(), res)

    # Create DEM xarray
    grid_dem_z = numpy.empty((1, len(dim_y), len(dim_x)), dtype=raster_type)
    dem = xarray.DataArray(grid_dem_z, coords={'band': [1], 'y': dim_y, 'x': dim_x}, dims=['band', 'y', 'x'],
                           attrs={'scale_factor': 1.0, 'add_offset': 0.0, 'long_name': 'idw'})
    dem.rio.write_crs(h_crs, inplace=True)
    dem.name = 'z'
    dem = dem.rio.write_nodata(numpy.nan)
    dem.data[0] = numpy.nan

    # Python IDW scipy - function
    start_time = time.time()
    z_out_flat = idw_function(xy_in, z_in, grid_xy, leafsize=leafsize, search_radius=radius, power=power, eps=eps)
    z_out = z_out_flat.reshape(grid_x.shape)
    end_time = time.time()
    idw_time = end_time-start_time
    if(verbose):
        print(f"IDW scipy function takes {idw_time}")
    dem.data[0] = z_out

    # Create roughness xarray
    xy_in = numpy.empty((len(all_points), 2))
    xy_in[:, 0] = all_points['X']
    xy_in[:, 1] = all_points['Y']
    z_in = all_points['Z']

    # Create roughness xarray
    voxel_height = 0.5
    dim_z = numpy.arange(0, 2, voxel_height) + voxel_height / 2
    voxel_heights = numpy.arange(0, 2 + voxel_height, voxel_height)
    grid_dem_r = numpy.empty((len(dim_z), len(dim_y), len(dim_x)), dtype=raster_type)
    roughness = xarray.DataArray(grid_dem_r, coords={'z': dim_z, 'y': dim_y, 'x': dim_x}, dims=['z', 'y', 'x'],
                                 attrs={'scale_factor': 1.0, 'add_offset': 0.0, 'long_name': 'vegetation density'})
    roughness.rio.write_crs(h_crs, inplace=True)
    roughness.name = 'w'
    roughness = roughness.rio.write_nodata(numpy.nan)
    roughness.data[0] = numpy.nan
    roughness_gs = roughness.copy(deep=True)

    # Python roughness - function Mewis
    start_time = time.time()
    rough_flat = roughness_function_mewis(xy_in, z_in, grid_xy, ground_elevations=z_out_flat, voxel_heights=voxel_heights,
                                          leafsize=leafsize, search_radius=radius, eps=eps)
    r_out = rough_flat.transpose().reshape(grid_dem_r.shape)
    end_time = time.time()
    rough_time = end_time-start_time
    if(verbose):
        print(f"Roughness function mewis takes {rough_time}")
    roughness.data = r_out

    # Python roughness - function Mewis
    start_time = time.time()
    rough_flat = roughness_function_graham(xy_in, z_in, grid_xy, ground_elevations=z_out_flat, voxel_heights=voxel_heights,
                                           leafsize=leafsize, search_radius=radius, eps=eps)
    r_out = rough_flat.transpose().reshape(grid_dem_r.shape)
    end_time = time.time()
    rough_time = end_time-start_time
    if(verbose):
        print(f"Roughness function graham takes {rough_time}")
    roughness_gs.data = r_out

    if(verbose):
        print("Plot results")
    vmin = roughness.data.min()
    vmax = roughness.data.max()
    roughness.isel(z=0).plot(vmin=vmin, vmax=vmax)
    roughness.isel(z=1).plot(vmin=vmin, vmax=vmax)
    roughness.isel(z=2).plot(vmin=vmin, vmax=vmax)
    roughness.isel(z=3).plot(vmin=vmin, vmax=vmax)
    dem.plot()
    vmin = max(vmin, 0.01)
    roughness.isel(z=0).plot(vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm())
    roughness.isel(z=1).plot(vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm())
    roughness.isel(z=2).plot(vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm())
    roughness.isel(z=3).plot(vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm())
    roughness_gs.isel(z=0).plot()
    roughness_gs.isel(z=1).plot()
    roughness_gs.isel(z=2).plot()
    roughness_gs.isel(z=3).plot()

if __name__ == "__main__":
    main_compare()
