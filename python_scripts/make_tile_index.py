# -*- coding: utf-8 -*-
"""
Run setup - download needed LiDAR files
"""

import argparse
import pathlib
import os
import geopandas
import shutil


def parse_args():
    """Expect a command line arguments of the form:
    '--output_path path_to_output_folder
     --tile_id tile_id_string'"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--las_path",
        metavar="str",
        required=True,
        action="store",
        help="the las path string - The path to the folder where the LAS files to index are.",
    )

    parser.add_argument(
        "--crs",
        metavar="str",
        required=True,
        action="store",
        help="the EPSG CRS int - The EPSG CRS int code.",
    )

    return parser.parse_args()


def main(las_path: str, crs: int):
    """ The make_tile_index.main Creates a TileIndex file for the las files in the 
        las_path folder. This has a 'name' colume with the las file name. """

    print("Run make_tile_index!")

    ## The LAS folder path
    las_path = pathlib.Path(las_path)
    
    ## Define the name of the index file
    index_name = f"{las_path.stem}_TileIndex"
    
    ## Run the shell command 
    os.system(f"find {str(las_path)} -iname '*.laz' | pdal tindex  create --tindex {str(las_path/index_name)} --lyr_name {index_name} "
              f"--stdin --t_srs EPSG:{crs} --filters.hexbin.edge_size=10 --filters.hexbin.threshold=1")

    
    ## Open and add a name column to the index
    tile_index = geopandas.read_file(las_path / index_name)
    tile_index['name'] = tile_index.apply(lambda row: pathlib.Path(row['location']).name, axis=1)
    tile_index.to_file(las_path / index_name / f"{index_name}.shp")
    
    ## Zip the index
    shutil.make_archive(index_name, 'zip', las_path / index_name)
    
    
    print("Finished!")


if __name__ == "__main__":
    """ If called as script: Read in the args and launch the main function"""
    args = parse_args()
    main(las_path=args.las_path, crs=args.crs)