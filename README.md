# Hydrologic-DEMs-scripts
Scipts associated with various aspects of generating hydrologically conditioned DEMs from LiDAR and other infrastructure data.

Note that anyone is welcome to use these scripts to tackle their own challenges - but these are not written with other peoples use intented. There is limited documentation and the data filepaths are hardcoded.

# Setting up a virtual environment
You can use - dem_generation.yml to create a conda virual environment that includes all the packages used by the scripts under the 'jupyter_notebooks' folder. See [link](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for instructions for creating an environment from a YML file. 

# Example script for creating a DEM
These are contained in the 'jupyter_notebooks' folder. They currently reference data on my local computer - you will need to either download your own data and update the file paths or figure out how to pull data from data access APIs. The two best example scripts are:
* [Waikanae_small_test_data_to_dem_separate_smooth_offshore.ipynb](https://github.com/rosepearson/Hydrologic-DEMs-scripts/blob/main/jupyter_notebooks/Waikanae_small_test_data_to_dem_separate_smooth_offshore.ipynb) - create a DEM with smooth RBF interpolation offshore (where there is sparse data). The produces various figures comparing different interpolation approaches.
* [Waikanae_small_test_data_to_dem_separate_offshore.ipynb](https://github.com/rosepearson/Hydrologic-DEMs-scripts/blob/main/jupyter_notebooks/Waikanae_small_test_data_to_dem_separate_offshore.ipynb) - create a DEM with linear interpolation offshore using PDAL functions.
