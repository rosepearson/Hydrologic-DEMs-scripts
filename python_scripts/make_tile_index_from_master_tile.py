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
        "--laz_path",
        metavar="str",
        required=True,
        action="store",
        help="the las path string - The path to the folder where the LAS files to index are.",
    )

    parser.add_argument(
        "--master_tile_index_file",
        metavar="str",
        required=True,
        action="store",
        help="the path to the master tile index file.",
    )

    return parser.parse_args()


def main(laz_path: str, master_tile_index_file: str):
    """ The make_tile_index.main Creates a TileIndex file for the las files in the 
        las_path folder. This has a 'file_name' column with the las file name. """

    print("Run make_tile_index_from_master_tile!")

    ## The LAS folder path
    laz_path = pathlib.Path(laz_path)
    master_tile = geopandas.read_file(master_tile_index_file)
    
    ## Define the name of the index file
    master_tile = master_tile[master_tile.apply(lambda row: (laz_path / f"{row['tilename']}.laz").exists(), axis=1)]
    master_tile["file_name"] = master_tile.apply(lambda row: f"{row['tilename']}.laz", axis=1)
    
    ## Save the tile index file
    master_tile.to_file(laz_path / f"{laz_path.stem}_TileIndex.gpkg")
    
    
    print("Finished!")


if __name__ == "__main__":
    """ If called as script: Read in the args and launch the main function"""
    args = parse_args()
    main(laz_path=args.laz_path, master_tile_index_file=args.master_tile_index_file)