import pandas as pd
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
from pathlib import Path
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


def get_MIP(experiment):
    """
    Utility function to get teh activity associated with a particular experiment
    """
    if experiment == 'ssp245-covid':
        return 'DAMIP'
    elif experiment == 'ssp370-lowNTCF':
        return 'AerChemMIP'
    elif experiment.startswith('ssp'):
        return 'ScenarioMIP'
    elif experiment.startswith('hist-'):
        return 'DAMIP'
    else:
        return 'CMIP'



def get_data(variable, experiment, member, frequency):
    """
    Helper function to get data, returns Xarray.dataset
    """
    try:
        base_path = Path(f"/glade/collections/cmip/CMIP6/{get_MIP(experiment)}/NCC/NorESM2-LM/")
        
        # Check if the experiment exists
        experiment_path = base_path / experiment
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment '{experiment}' not found at {experiment_path}")

        # Check if the member exists
        member_path = experiment_path / member
        if not member_path.exists():
            raise FileNotFoundError(f"Member '{member}' not found in {experiment_path}")

        # Check for the variable in the specified frequency
        data_path = member_path / frequency / variable / 'gn'
        
        if not data_path.exists():
            raise FileNotFoundError(f"Variable '{variable}' not found in '{frequency}'")

        # Find the latest version of the dataset
        try:
            versions = [f for f in data_path.iterdir() if f.is_dir()]
            if not versions:
                raise FileNotFoundError(f"No versions found for variable '{variable}' in {data_path}")
            versions.sort(reverse=True)
            latest_version = versions[0]
        except Exception as e:
            print(f"Error finding versions: {e}")

        # Get the path to the variable's files
        file_path = latest_version / variable
        nc_files = list(file_path.glob("*.nc"))
        
        if not nc_files:
            raise FileNotFoundError(f"No NetCDF files found for variable '{variable}' in {file_path}")
        
        # Load the dataset using xarray
        return xr.open_mfdataset(nc_files)[variable]
    
    except FileNotFoundError as e:
        print(f"File not found error: {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




def get_daily_data(variable, experiment, member):
    """
    Wrapper function to get daily data.
    """
    return get_data(variable, experiment, member, 'day')



def get_monthly_data(variable, experiment, member):
    """
    Wrapper function to get monthly data.
    """
    return get_data(variable, experiment, member, 'Amon')



# List of experiments and variables we are interested in
experiments = [
               '1pctCO2', 'abrupt-4xCO2', 'historical', 'piControl', # CMIP
               'hist-GHG', 'hist-aer'# DAMIP
               'ssp126', 'ssp245', 'ssp370', 'ssp370-lowNTCF', 'ssp585' #	ScenarioMIP
]

variables = [
             'tas', 'tasmin', 'tasmax', 'pr'
]



def save_netcdf():
    for exp in experiments:
        for r in range(1, 4):
            member = f"r{r}i1p1f1"

            # Initialize a dictionary to hold the data for each variable
            ds_dict = {}
            
            # Try to get precipitation data ('pr')
            pr = get_monthly_data('pr', exp, member)
            if pr is not None:
                pr = pr.chunk({'time': -1}).persist()
                ds_dict['pr'] = pr.groupby('time.year').mean('time')
                # Add the 90th percentile of precipitation
                ds_dict['pr90'] = pr.groupby('time.year').quantile(0.9, skipna=True)
            else:
                print(f"No data for 'pr' in experiment '{exp}' with member '{member}'")
            
            # Get temperature data ('tas')
            tas = get_monthly_data('tas', exp, member)
            if tas is not None:
                ds_dict['tas'] = tas.groupby('time.year').mean('time')
            else:
                print(f"No data for 'tas' in experiment '{exp}' with member '{member}'")
            
            # Get minimum and maximum temperature data ('tasmin', 'tasmax')
            tasmin = get_daily_data('tasmin', exp, member)
            tasmax = get_daily_data('tasmax', exp, member)
            if tasmin is not None and tasmax is not None:
                dtr = tasmax - tasmin
                ds_dict['diurnal_temperature_range'] = dtr.groupby('time.year').mean('time')
            else:
                print(f"Cannot get 'dtr' in experiment '{exp}' with member '{member}'")
            
            # If there are any variables successfully loaded, save the dataset
            if ds_dict:
                print(f'Writing NorESM2-LM_{exp}_{member}.nc')
                ds = xr.Dataset(ds_dict)
                ds.to_netcdf(Path.home()/'data'/f"NorESM2-LM_{exp}_{member}.nc")
            else:
                print(f"No valid data found for experiment '{exp}' with member '{member}', skipping...")

    print('All tasks finished')



if __name__ == '__main__':
    save_netcdf()