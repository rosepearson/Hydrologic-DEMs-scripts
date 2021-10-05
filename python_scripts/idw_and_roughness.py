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
    std_out = numpy.zeros(len(xy_out))

    for i, (near_indicies, point) in enumerate(zip(tree_index_list, xy_out)):

        if len(near_indicies) == 0:  # Set NaN if no values in search region
            z_out[i] = numpy.nan
            std_out[i] = numpy.nan
        else:
            distance_vectors = point - tree.data[near_indicies]
            smoothed_distances = numpy.sqrt(((distance_vectors**2).sum(axis=1)+smoothing**2))
            if smoothed_distances.min() == 0:  # incase the of an exact match
                z_out[i] = z_in[tree.query(point, k=1)[1]]
            else:
                z_out[i] = (z_in[near_indicies] / (smoothed_distances**power)).sum(axis=0) \
                    / (1 / (smoothed_distances**power)).sum(axis=0)
            std_out[i] = z_in[near_indicies].std()
    return z_out, std_out


def combined_roughness_function_mewis(all_points: numpy.ndarray, gnd_points: numpy.ndarray, xy_out: numpy.ndarray,
                                      search_radius: float, voxel_heights: numpy.ndarray, leafsize: int = 10,
                                      eps: float = 0, smoothing: float = 0, power: int = 1):

    xy_in_gnd = numpy.empty((len(gnd_points), 2))
    xy_in_gnd[:, 0] = gnd_points['X']
    xy_in_gnd[:, 1] = gnd_points['Y']
    z_in_gnd = gnd_points['Z']

    tree_gnd = scipy.spatial.KDTree(xy_in_gnd, leafsize=leafsize)  # build the tree
    tree_index_list_gnd = tree_gnd.query_ball_point(xy_out, r=search_radius, eps=eps)  # , eps=0.2)
    z_out_gnd = numpy.zeros(len(xy_out))

    for i, (near_indicies, point) in enumerate(zip(tree_index_list_gnd, xy_out)):

        if len(near_indicies) == 0:  # Set NaN if no values in search region
            z_out_gnd[i] = numpy.nan
        else:
            distance_vectors = point - tree_gnd.data[near_indicies]
            smoothed_distances = numpy.sqrt(((distance_vectors**2).sum(axis=1)+smoothing**2))
            if smoothed_distances.min() == 0:  # incase the of an exact match
                z_out_gnd[i] = z_in_gnd[tree_gnd.query(point, k=1)[1]]
            else:
                z_out_gnd[i] = (z_in_gnd[near_indicies] / (smoothed_distances**power)).sum(axis=0) \
                    / (1 / (smoothed_distances**power)).sum(axis=0)

    ground_elevations = z_out_gnd

    xy_in_all = numpy.empty((len(all_points), 2))
    xy_in_all[:, 0] = all_points['X']
    xy_in_all[:, 1] = all_points['Y']
    z_in_all = all_points['Z']

    tree_all = scipy.spatial.KDTree(xy_in_all, leafsize=leafsize)  # build the tree
    tree_index_list_all = tree_all.query_ball_point(xy_out, r=search_radius, eps=eps)
    z_out_all = numpy.empty((len(xy_out), len(voxel_heights) - 1))

    for i, near_indicies in enumerate(tree_index_list_all):

        # count number of points in each height
        near_indicies = numpy.array(near_indicies)
        near_heights = z_in_all[near_indicies]
        count_below_height = numpy.array([len(near_indicies[near_heights < threshold_elevation])
                                          for threshold_elevation in voxel_heights + ground_elevations[i]])
        z_out_all[i] = numpy.log(count_below_height[1:] / count_below_height[:-1]) \
            / (voxel_heights[1:] - voxel_heights[:-1])

    return z_out_gnd, z_out_all


def roughness_function_mewis(xy_in: numpy.ndarray, z_in: numpy.ndarray, xy_out: numpy.ndarray, search_radius: float,
                             ground_elevations: numpy.ndarray, ground_std: numpy.ndarray,
                             voxel_heights: numpy.ndarray, leafsize: int = 10, eps: float = 0):
    assert len(xy_in) == len(z_in), f"len(X) of {len(xy_in)} != len(z_in) {len(z_in)}"
    assert type(z_in) is numpy.ndarray and z_in.ndim == 1, f"Expecting z_in to be a 1D numpy array instead got {z_in}"
    assert type(xy_out) is numpy.ndarray and xy_out.ndim == 2, "Expect xy_out to be a 2D numpy array " \
        f" instead got {xy_out}"
    #assert voxel_heights[-1,0] > voxel_heights[0], "The voxel heights must be monotonically increasing"

    tree = scipy.spatial.KDTree(xy_in, leafsize=leafsize)  # build the tree
    tree_index_list = tree.query_ball_point(xy_out, r=search_radius, eps=eps)
    r_out = numpy.empty((len(xy_out), len(voxel_heights) - 1))

    ground_voxel_height = voxel_heights[1] - voxel_heights[0]

    for i, near_indicies in enumerate(tree_index_list):

        # count number of points in each height
        near_indicies = numpy.array(near_indicies)
        near_heights = z_in[near_indicies]
        ground_voxel_heights = voxel_heights + ground_elevations[i]
        '''if ground_std[i] > 10 * ground_voxel_height:
            ground_voxel_heights[0] = ground_elevations[i] - ground_std[i] / 2'''
        count_below_height = numpy.array([len(near_indicies[near_heights < threshold_elevation])
                                          for threshold_elevation in ground_voxel_heights])
        r_out[i] = numpy.log(count_below_height[1:] / count_below_height[:-1]) \
            / (voxel_heights[1:] - voxel_heights[:-1])

    return r_out


def mewis_n_function(veg_density: xarray.DataArray, depth: xarray.DataArray, voxel_total_heights: numpy.ndarray,
                     minimum_height: float = 20, drag_coefficient: float = 1.2):
    assert numpy.all(veg_density.x == depth.x), "Expect the same x grids for veg_density and depth."
    assert numpy.all(veg_density.y == depth.y), "Expect the same y grids for veg_density and depth."

    '''# Calculate lambda (The Darcy-Weisbach friction coefficient)
    lambda_coefficient = depth.copy(deep=True)
    lambda_coefficient.data[:]=0
    lambda_coefficient = lambda_coefficient.assign_attrs({'long_name':"Lambda the Darcy-Weisbach friction coefficient"})
    lambda_coefficient.name = 'lambda'

    for i in range(len(veg_density.z)):
        lambda_coefficient.data[depth.data > voxel_total_heights[i + 1]] += 4 * drag_coefficient * veg_density.data[i, depth.data > voxel_total_heights[i + 1]] \
            * (voxel_total_heights[i + 1] - voxel_total_heights[i])
        lambda_coefficient.data[(depth.data > voxel_total_heights[i]) & (depth.data < voxel_total_heights[i + 1])] += 4 * drag_coefficient \
            * veg_density.data[i, (depth.data > voxel_total_heights[i]) & (depth.data < voxel_total_heights[i + 1])] \
                * (depth.data[(depth.data > voxel_total_heights[i]) & (depth.data < voxel_total_heights[i + 1])] - voxel_total_heights[i])

    fig, ax = matplotlib.pyplot.subplots()
    lambda_coefficient.plot()
    ax.set_title(f"Plot: Lambda calculated from the Mewis equations")

    # Calculate k_st (The Manning-Strickler coefficient)
    kst_coefficient = depth.copy(deep=True)
    kst_coefficient = kst_coefficient.assign_attrs({'long_name':"The Manning-Strickler coefficient"})
    kst_coefficient.name = 'k_st'
    kst_coefficient.data = numpy.sqrt(8*9.8/(lambda_coefficient.data*numpy.cbrt(depth.data)))

    fig, ax = matplotlib.pyplot.subplots()
    kst_coefficient.plot(norm=matplotlib.colors.LogNorm())
    ax.set_title(f"Plot: K_st calculated from the Mewis equations")

    # Calculate Manning's n - the inverse of the Manning-Strickler coefficient
    n_coefficient = depth.copy(deep=True)
    n_coefficient = n_coefficient.assign_attrs({'long_name':"The Manning's n coefficient"})
    n_coefficient.name = 'n'
    n_coefficient.data = 1/kst_coefficient.data'''

    # Calculate Manning's n
    n_coefficient = depth.copy(deep=True)
    n_coefficient.data[:] = 0
    n_coefficient = n_coefficient.assign_attrs({'long_name': "The Manning's n coefficient"})
    n_coefficient.name = 'n'

    # Calculate lambda (The Darcy-Weisbach friction coefficient) using the Mewis paper equation
    for i in range(len(veg_density.z)):
        n_coefficient.data[depth.data > voxel_total_heights[i + 1]] += 4 * drag_coefficient * veg_density.data[i, depth.data > voxel_total_heights[i + 1]] \
            * (voxel_total_heights[i + 1] - voxel_total_heights[i])
        n_coefficient.data[(depth.data > voxel_total_heights[i]) & (depth.data < voxel_total_heights[i + 1])] += 4 * drag_coefficient \
            * veg_density.data[i, (depth.data > voxel_total_heights[i]) & (depth.data < voxel_total_heights[i + 1])] \
                * (depth.data[(depth.data > voxel_total_heights[i]) & (depth.data < voxel_total_heights[i + 1])] - voxel_total_heights[i])

    # Convert to Manning's n
    n_coefficient.data = numpy.sqrt(n_coefficient.data * numpy.cbrt(depth.data) / (8 * 9.8))

    return n_coefficient


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
    z_out_flat, z_std_flat = idw_function(xy_in, z_in, grid_xy, leafsize=leafsize, search_radius=radius,
                                          power=power, eps=eps)
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
    dim_z = numpy.arange(0, 2, voxel_height) + voxel_height / 2  # middle of cell
    voxel_cell_heights = numpy.ones(len(dim_z)) * voxel_height  # height of each voxel
    voxel_total_heights = numpy.concatenate([dim_z - voxel_cell_heights / 2,
                                            [dim_z[-1] + voxel_cell_heights[-1] / 2]])  # bottom to top of each voxel
    print(f"Voxel bin middles is {dim_z}, cell_heights are {voxel_cell_heights}, and voxel bottoms and tops are {voxel_total_heights}")
    grid_dem_r = numpy.empty((len(dim_z), len(dim_y), len(dim_x)), dtype=raster_type)
    veg_density = xarray.DataArray(grid_dem_r, coords={'z': dim_z, 'y': dim_y, 'x': dim_x, 'h': ('z', voxel_cell_heights)},
                                   dims=['z', 'y', 'x'],
                                   attrs={'scale_factor': 1.0, 'add_offset': 0.0, 'long_name': 'vegetation density'})
    veg_density.rio.write_crs(h_crs, inplace=True)
    veg_density.name = 'w'
    veg_density = veg_density.rio.write_nodata(numpy.nan)
    veg_density.data[0] = numpy.nan
    roughness_gs = veg_density.copy(deep=True)

    # Examine why there are differences in the gnd height and all points
    '''z_gnd, z_all = combined_roughness_function_mewis(all_points, gnd_points, grid_xy, voxel_heights=voxel_total_heights,
                                                     leafsize=leafsize, search_radius=radius, power=power, eps=eps)'''

    # Python roughness - function Mewis - veg_density
    start_time = time.time()
    rough_flat = roughness_function_mewis(xy_in, z_in, grid_xy, ground_elevations=z_out_flat, ground_std=z_std_flat,
                                          voxel_heights=voxel_total_heights, leafsize=leafsize,
                                          search_radius=radius, eps=eps)
    r_out = rough_flat.transpose().reshape(grid_dem_r.shape)
    end_time = time.time()
    rough_time = end_time-start_time
    if(verbose):
        print(f"Roughness function mewis takes {rough_time}")
    veg_density.data = r_out

    # Python roughness - function Mewis
    start_time = time.time()
    rough_flat = roughness_function_graham(xy_in, z_in, grid_xy, ground_elevations=z_out_flat,
                                           voxel_heights=voxel_total_heights, leafsize=leafsize,
                                           search_radius=radius, eps=eps)
    r_out = rough_flat.transpose().reshape(grid_dem_r.shape)
    end_time = time.time()
    rough_time = end_time-start_time
    if(verbose):
        print(f"Roughness function graham takes {rough_time}")
    roughness_gs.data = r_out

    if(verbose):
        print("Plot results")
    vmin = veg_density.data.min()
    vmax = veg_density.data.max()
    for i in range(len(dim_z)):
        fig, ax = matplotlib.pyplot.subplots()
        veg_density.isel(z=i).plot(vmin=vmin, vmax=vmax)
        ax.set_title(f"Plot: vegetation density bin {voxel_total_heights[i]}-{voxel_total_heights[i+1]}m")

    fig, ax = matplotlib.pyplot.subplots()
    dem.plot()
    ax.set_title('Plot: DEM')

    vmin = max(vmin, 0.01)
    for i in range(len(dim_z)):
        fig, ax = matplotlib.pyplot.subplots()
        veg_density.isel(z=i).plot(vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm())
        ax.set_title(f"Log plot: vegetation density bin {voxel_total_heights[i]}-{voxel_total_heights[i+1]}m")

    vmin = roughness_gs.data.min()
    vmax = roughness_gs.data.max()
    for i in range(len(dim_z)):
        fig, ax = matplotlib.pyplot.subplots()
        roughness_gs.isel(z=i).plot(vmin=vmin, vmax=vmax)
        ax.set_title(f"Plot: Graeme's coefficient bin {voxel_total_heights[i]}-{voxel_total_heights[i+1]}m")

    vmin = max(vmin, 0.01)
    for i in range(len(dim_z)):
        fig, ax = matplotlib.pyplot.subplots()
        roughness_gs.isel(z=i).plot(vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm())
        ax.set_title(f"Log plot: Graeme's coefficient bin {voxel_total_heights[i]}-{voxel_total_heights[i+1]}m")

    return dem, veg_density


def main_manual_roughness(dem: xarray.DataArray):
    # load in The zo and manning's n files manaully generated by Graeme
    base_path = pathlib.Path(r'C:\Users\pearsonra\Documents\data\roughness\Waikanae_roughness')
    zo_file = base_path / r'Zo.asc'
    n_file_2cm_min_v2 = base_path / r'Waikanae_n_values.asc'

    zo = numpy.loadtxt(zo_file, skiprows=6)
    with open(zo_file) as f:
        n_cols = int(''.join(filter(str.isdigit, f.readline())))
        n_rows = int(''.join(filter(str.isdigit, f.readline())))
        x_min = float(''.join(filter(str.isdigit, f.readline())))
        y_min = float(''.join(filter(str.isdigit, f.readline())))
        res = float(f.readline().split()[1])
        no_data = float(f.readline().split()[1])
    dim_x = numpy.arange(x_min, x_min + n_cols * res, res)
    dim_y = numpy.arange(y_min, y_min + n_rows * res, res)

    dim_x_mask = (dim_x > float(dem.x.min())) & (dim_x < float(dem.x.max()))
    dim_y_mask = (dim_y > float(dem.y.min())) & (dim_y < float(dem.y.max()))

    zo_clipped = zo[dim_y_mask, :]
    zo_clipped = zo_clipped[:, dim_x_mask]

    zo_array = xarray.DataArray(zo_clipped, coords={'y': dim_y[dim_y_mask], 'x': dim_x[dim_x_mask]}, dims=['y', 'x'],
                                attrs={'scale_factor': 1.0, 'add_offset': 0.0, 'long_name': 'zo manual from Graeme'})
    zo_array.name = 'zo manual'
    zo_array = zo_array.rio.write_nodata(numpy.nan)
    zo_array.data[zo_array.data == no_data] = numpy.nan

    fig, ax = matplotlib.pyplot.subplots()
    zo_array.plot()
    ax.set_title("Plot: Zo as manually generated by Graeme")

    n_2cm_min_v2 = numpy.loadtxt(n_file_2cm_min_v2, skiprows=6)

    with open(n_file_2cm_min_v2) as f:
        n_cols = int(''.join(filter(str.isdigit, f.readline())))
        n_rows = int(''.join(filter(str.isdigit, f.readline())))
        x_min = float(f.readline().split()[1])
        y_min = float(f.readline().split()[1])
        res = float(f.readline().split()[1])
        no_data = float(f.readline().split()[1])
    dim_x = numpy.arange(x_min, x_min + n_cols * res, res)
    dim_y = numpy.arange(y_min + n_rows * res, y_min, -res)

    dim_x_mask = (dim_x > float(dem.x.min())) & (dim_x < float(dem.x.max()))
    dim_y_mask = (dim_y > float(dem.y.min())) & (dim_y < float(dem.y.max()))

    n_clipped = n_2cm_min_v2[dim_y_mask, :]
    n_clipped = n_clipped[:, dim_x_mask]

    n_array = xarray.DataArray(n_clipped, coords={'y': dim_y[dim_y_mask], 'x': dim_x[dim_x_mask]}, dims=['y', 'x'],
                               attrs={'scale_factor': 1.0, 'add_offset': 0.0,
                                      'long_name': "Manning's n' manual from Graeme"})
    n_array.name = 'n manual'
    n_array = n_array.rio.write_nodata(numpy.nan)
    n_array.data[n_array.data == no_data] = numpy.nan

    fig, ax = matplotlib.pyplot.subplots()
    n_array.plot(norm=matplotlib.colors.LogNorm())
    ax.set_title("Plot: Manning's n calculated from Graeme's manually generated Zo")


def main_calculate_mannings_n(veg_density: xarray.DataArray):
    # Calculate the manning'n by integrating by flood depth
    minimum_height = 0.2  # in m
    drag_coefficient = 1.2

    base_path = pathlib.Path(r'C:\Users\pearsonra\Documents\data\roughness\Waikanae_roughness')
    depth_file = base_path / r'BGF_Rivers+ROG_hmax_resampled-wide.asc'  # .nc

    # Load in the depth file
    '''with rioxarray.rioxarray.open_rasterio(depth_file, masked=True) as depth: 
        depth.load()
    depth = depth.copy(deep=True)  # Note this reads in with actual_range for x & y dims - causing issues'''
    # manually load in depth file
    depth = numpy.loadtxt(depth_file, skiprows=6)

    with open(depth_file) as f:
        n_cols = int(f.readline().split()[1])
        n_rows = int(f.readline().split()[1])
        x_min = float(f.readline().split()[1])
        y_min = float(f.readline().split()[1])
        res = float(f.readline().split()[1])
        no_data = float(f.readline().split()[1])
    dim_x = numpy.arange(x_min, x_min + n_cols * res, res)
    dim_y = numpy.arange(y_min + n_rows * res, y_min, -res)

    depth = xarray.DataArray(depth, coords={'y': dim_y, 'x': dim_x}, dims=['y', 'x'],
                             attrs={'scale_factor': 1.0, 'add_offset': 0.0, 'long_name': "Maximum water depths from BG-FLOOD"})
    depth.name = 'hmax'
    depth = depth.rio.write_nodata(numpy.nan)
    depth.data[depth.data == no_data] = numpy.nan

    dim_x_mask = (dim_x > float(veg_density.x.min())) & (dim_x < float(veg_density.x.max()))
    dim_y_mask = (dim_y > float(veg_density.y.min())) & (dim_y < float(veg_density.y.max()))

    depth_clipped = depth[dim_y_mask, :]
    depth_clipped = depth_clipped[:, dim_x_mask]

    depth_clipped = xarray.DataArray(depth_clipped, coords={'y': dim_y[dim_y_mask], 'x': dim_x[dim_x_mask]},
                                     dims=['y', 'x'], attrs={'scale_factor': 1.0, 'add_offset': 0.0,
                                                             'long_name': "Maximum water depths from BG-FLOOD"})
    depth_clipped.name = 'hmax'
    depth_clipped = depth_clipped.rio.write_nodata(numpy.nan)
    depth_clipped.data[depth_clipped.data == no_data] = numpy.nan

    # Give a minimum value and resample the depths onto the same grid as the veg_density
    depth_aligned_with_min = depth_clipped.copy(deep=True)
    depth_aligned_with_min.data[numpy.isnan(depth_aligned_with_min.data)] = minimum_height
    depth_aligned_with_min = depth_aligned_with_min.interp_like(veg_density.mean('z'))
    depth_aligned_with_min.data[numpy.isnan(depth_aligned_with_min.data)] = minimum_height

    # calculate the array of voxel heights
    voxel_height = 0.5
    dim_z = numpy.arange(0, 2.5, voxel_height)
    voxel_cell_heights = numpy.ones(len(dim_z)) * voxel_height
    voxel_total_heights = numpy.concatenate([dim_z-voxel_cell_heights / 2, [dim_z[-1] + voxel_cell_heights[-1] / 2]])

    # Calculate lambda from the mewis paper and then mannings n
    mannings_n_mewis = mewis_n_function(veg_density, depth_aligned_with_min, voxel_total_heights,
                                        minimum_height=minimum_height, drag_coefficient=drag_coefficient)

    fig, ax = matplotlib.pyplot.subplots()
    depth_clipped.plot()
    ax.set_title("Plot: Depth as reported from BG-FLOOD")

    depth_min = depth_clipped.copy(deep=True)
    depth_min.data[numpy.isnan(depth_min.data)] = minimum_height
    fig, ax = matplotlib.pyplot.subplots()
    depth_min.plot(vmin=0.1, vmax=1)
    ax.set_title("Plot: Depth after 20cm added as Graham did")

    vmin = max(mannings_n_mewis.data.min(), 0.01)
    vmax = mannings_n_mewis.data.max()
    fig, ax = matplotlib.pyplot.subplots()
    mannings_n_mewis.plot(vmin=vmin, vmax=vmax, norm=matplotlib.colors.LogNorm())
    ax.set_title("Plot: Manning's n calculated from the Mewis equations")


if __name__ == "__main__":
    dem, veg_density = main_compare()
    main_manual_roughness(dem)
    main_calculate_mannings_n(veg_density)
