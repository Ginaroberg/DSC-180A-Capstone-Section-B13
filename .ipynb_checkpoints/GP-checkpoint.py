# imports
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from esem import gp_model
from eofs.xarray import Eof
from utils import *
import gpflow
import os



# List of experiment data used for training
train_files= ['ssp126', 'ssp370', 'ssp585', 'historical', 'hist-GHG', 'hist-aer']

# Create the folder to save outputs
os.makedirs('./outputs', exist_ok=True)




def emulate_tas():
    # Get data
    Xtrain, eof_solvers = create_predictor_data(train_files)
    Ytrain_tas = create_predictdand_data(train_files)['tas'].values.reshape(-1, 96*144)
    Xtest = get_test_data('ssp245', eof_solvers)
    Ytest = xr.open_dataset('./test/outputs_ssp245.nc').compute()
    tas_truth = Ytest['tas'].mean('member')
    
    # Drop rows including nans
    train_nan_mask = Xtrain.isna().any(axis=1).values
    Xtrain = Xtrain.dropna(axis=0, how='any')
    Ytrain_tas = Ytrain_tas[~train_nan_mask]
    assert Xtrain.shape[0]==Ytrain_tas.shape[0]
    
    test_nan_mask = Xtest.isna().any(axis=1).values
    Xtest = Xtest.dropna(axis=0, how='any')
    tas_truth = tas_truth[~test_nan_mask]
    
    # Standardize predictor fields requiring standardization (non-EOFs)
    train_CO2_mean, train_CO2_std = Xtrain['CO2'].mean(), Xtrain['CO2'].std()
    train_CH4_mean, train_CH4_std = Xtrain['CH4'].mean(), Xtrain['CH4'].std()
    Xtrain['CO2'] = (Xtrain['CO2'] - train_CO2_mean) / train_CO2_std
    Xtrain['CH4'] = (Xtrain['CH4'] - train_CH4_mean) / train_CH4_std
    Xtest['CO2'] = (Xtest['CO2'] - train_CO2_mean) / train_CO2_std
    Xtest['CH4'] = (Xtest['CH4'] - train_CH4_mean) / train_CH4_std
    
    # Standardize predictand fields
    train_tas_mean, train_tas_std = Ytrain_tas.mean(), Ytrain_tas.std()
    Ytrain_tas = (Ytrain_tas - train_tas_mean) / train_tas_std

    # Define kernels
    kernel_CO2 = gpflow.kernels.Matern32(active_dims=[0]) # active_dims specifies which dimension the kernel is applied to
    kernel_CH4 = gpflow.kernels.Matern32(active_dims=[1])
    kernel_BC = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[2, 3, 4, 5, 6])
    kernel_SO2 = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[7, 8, 9, 10, 11])
    kernel = kernel_CO2 + kernel_CH4 + kernel_BC + kernel_SO2

    # Define Model
    mean = gpflow.mean_functions.Constant()
    model = gpflow.models.GPR(data=(Xtrain.astype(np.float64), # cast to float64 because gpflow requires numerical stability
                                    Ytrain_tas.astype(np.float64)),
                              kernel = kernel,
                              mean_function = mean)
    
    # Define optimizer
    optimizer = gpflow.optimizers.Scipy()
    
    # Train
    optimizer.minimize(model.training_loss,
                       variables=model.trainable_variables,
                       options=dict(disp=True, maxiter=1000))
    
    # Predict
    standard_posterior_mean, standard_posterior_var = model.predict_y(Xtest.values) # predicted mean of GP, predicted variance of GP
    posterior_mean = standard_posterior_mean * train_tas_std + train_tas_mean # transform mean prediction to original scale
    posterior_stddev = np.sqrt(standard_posterior_var) * train_tas_std # transform variance prediction to original scale standard deviation

    # Put output back into xarray format
    posterior_tas = np.reshape(posterior_mean, [86, 96, 144])
    posterior_tas_stddev = np.reshape(posterior_stddev, [86, 96, 144])
    
    posterior_tas_data = xr.DataArray(posterior_tas, dims=tas_truth.dims, coords=tas_truth.coords)
    posterior_tas_std_data = xr.DataArray(posterior_tas_stddev, dims=tas_truth.dims, coords=tas_truth.coords)

    # Save prediction
    posterior_tas_data.to_netcdf('./outputs/GP-posterior-mean-tas-test-2019-2100.nc')
    posterior_tas_std_data.to_netcdf('./outputs/GP-posterior-std-tas-test-2019-2100.nc')


def emulate_pr():
    # Get data
    Xtrain, eof_solvers = create_predictor_data(train_files)
    Ytrain_pr = create_predictdand_data(train_files)['pr'].values.reshape(-1, 96*144)
    Xtest = get_test_data('ssp245', eof_solvers)
    Ytest = xr.open_dataset('./test/outputs_ssp245.nc').compute()
    pr_truth = 86400 * Ytest['pr'].mean('member') # convert pr to mm/day, default unit is kg/m²/s
    
    # Drop rows including nans
    train_nan_mask = Xtrain.isna().any(axis=1).values
    Xtrain = Xtrain.dropna(axis=0, how='any')
    Ytrain_pr = Ytrain_pr[~train_nan_mask]
    assert Xtrain.shape[0]==Ytrain_pr.shape[0]
    
    test_nan_mask = Xtest.isna().any(axis=1).values
    Xtest = Xtest.dropna(axis=0, how='any')
    pr_truth = pr_truth[~test_nan_mask]
    
    # Standardize predictor fields requiring standardization (non-EOFs)
    train_CO2_mean, train_CO2_std = Xtrain['CO2'].mean(), Xtrain['CO2'].std()
    train_CH4_mean, train_CH4_std = Xtrain['CH4'].mean(), Xtrain['CH4'].std()
    Xtrain['CO2'] = (Xtrain['CO2'] - train_CO2_mean) / train_CO2_std
    Xtrain['CH4'] = (Xtrain['CH4'] - train_CH4_mean) / train_CH4_std
    Xtest['CO2'] = (Xtest['CO2'] - train_CO2_mean) / train_CO2_std
    Xtest['CH4'] = (Xtest['CH4'] - train_CH4_mean) / train_CH4_std
    
    # Standardize predictand fields
    train_pr_mean, train_pr_std = Ytrain_pr.mean(), Ytrain_pr.std()
    Ytrain_pr = (Ytrain_pr - train_pr_mean) / train_pr_std

    # Define kernels
    kernel_CO2 = gpflow.kernels.Matern32(active_dims=[0]) # active_dims specifies which dimension the kernel is applied to
    kernel_CH4 = gpflow.kernels.Matern32(active_dims=[1])
    kernel_BC = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[2, 3, 4, 5, 6])
    kernel_SO2 = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[7, 8, 9, 10, 11])
    kernel = kernel_CO2 + kernel_CH4 + kernel_BC + kernel_SO2
    
    # Define Model
    mean = gpflow.mean_functions.Constant()
    model = gpflow.models.GPR(data=(Xtrain.astype(np.float64), # cast to float64 because gpflow requires numerical stability
                                    Ytrain_pr.astype(np.float64)),
                              kernel = kernel,
                              mean_function = mean)
    
    # Define optimizer
    optimizer = gpflow.optimizers.Scipy()
    
    # Train
    optimizer.minimize(model.training_loss,
                       variables=model.trainable_variables,
                       options=dict(disp=True, maxiter=1000))
    
    # Predict
    standard_posterior_mean, standard_posterior_var = model.predict_y(Xtest.values) # predicted mean of GP, predicted variance of GP
    posterior_mean = standard_posterior_mean * train_pr_std + train_pr_mean # transform mean prediction to original scale
    posterior_stddev = np.sqrt(standard_posterior_var) * train_pr_std # transform variance prediction to original scale standard deviation
    
    # Put output back into xarray format for calculating RMSE/plotting
    posterior_pr = np.reshape(posterior_mean, [86, 96, 144])
    posterior_pr_stddev = np.reshape(posterior_stddev, [86, 96, 144])
    
    posterior_pr_data = xr.DataArray(posterior_pr, dims=pr_truth.dims, coords=pr_truth.coords)
    posterior_pr_std_data = xr.DataArray(posterior_pr_stddev, dims=pr_truth.dims, coords=pr_truth.coords)
    
    # Save prediction
    posterior_pr_data.to_netcdf('./outputs/GP-posterior-mean-pr-test-2019-2100.nc')
    posterior_pr_std_data.to_netcdf('./outputs/GP-posterior-std-pr-test-2019-2100.nc')


def emulate_pr90():
    # Get data
    Xtrain, eof_solvers = create_predictor_data(train_files)
    Ytrain_pr90 = create_predictdand_data(train_files)['pr90'].values.reshape(-1, 96*144)
    Xtest = get_test_data('ssp245', eof_solvers)
    Ytest = xr.open_dataset('./test/outputs_ssp245.nc').compute()
    pr90_truth = 86400 * Ytest['pr90'].mean('member') # convert pr to mm/day, default unit is kg/m²/s

    # Drop rows including nans
    train_nan_mask = Xtrain.isna().any(axis=1).values
    Xtrain = Xtrain.dropna(axis=0, how='any')
    Ytrain_pr90 = Ytrain_pr90[~train_nan_mask]
    assert Xtrain.shape[0]==Ytrain_pr90.shape[0]

    test_nan_mask = Xtest.isna().any(axis=1).values
    Xtest = Xtest.dropna(axis=0, how='any')
    pr90_truth = pr90_truth[~test_nan_mask]

    # Standardize predictor fields requiring standardization (non-EOFs)
    train_CO2_mean, train_CO2_std = Xtrain['CO2'].mean(), Xtrain['CO2'].std()
    train_CH4_mean, train_CH4_std = Xtrain['CH4'].mean(), Xtrain['CH4'].std()
    Xtrain['CO2'] = (Xtrain['CO2'] - train_CO2_mean) / train_CO2_std
    Xtrain['CH4'] = (Xtrain['CH4'] - train_CH4_mean) / train_CH4_std
    Xtest['CO2'] = (Xtest['CO2'] - train_CO2_mean) / train_CO2_std
    Xtest['CH4'] = (Xtest['CH4'] - train_CH4_mean) / train_CH4_std

    # Standardize predictand fields
    train_pr90_mean, train_pr90_std = Ytrain_pr90.mean(), Ytrain_pr90.std()
    Ytrain_pr90 = (Ytrain_pr90 - train_pr90_mean) / train_pr90_std

    # Define kernels
    kernel_CO2 = gpflow.kernels.Matern32(active_dims=[0]) # active_dims specifies which dimension the kernel is applied to
    kernel_CH4 = gpflow.kernels.Matern32(active_dims=[1])
    kernel_BC = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[2, 3, 4, 5, 6])
    kernel_SO2 = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[7, 8, 9, 10, 11])
    kernel = kernel_CO2 + kernel_CH4 + kernel_BC + kernel_SO2

    # Define Model
    mean = gpflow.mean_functions.Constant()
    model = gpflow.models.GPR(data=(Xtrain.astype(np.float64), # cast to float64 because gpflow requires numerical stability
                                    Ytrain_pr90.astype(np.float64)),
                              kernel = kernel,
                              mean_function = mean)

    # Define optimizer
    optimizer = gpflow.optimizers.Scipy()

    # Train
    optimizer.minimize(model.training_loss,
                       variables=model.trainable_variables,
                       options=dict(disp=True, maxiter=1000))

    # Predict
    standard_posterior_mean, standard_posterior_var = model.predict_y(Xtest.values) # predicted mean of GP, predicted variance of GP
    posterior_mean = standard_posterior_mean * train_pr90_std + train_pr90_mean # transform mean prediction to original scale
    posterior_stddev = np.sqrt(standard_posterior_var) * train_pr90_std # transform variance prediction to original scale standard deviation

    # Put output back into xarray format for calculating RMSE/plotting
    posterior_pr90 = np.reshape(posterior_mean, [86, 96, 144])
    posterior_pr90_stddev = np.reshape(posterior_stddev, [86, 96, 144])
    
    posterior_pr90_data = xr.DataArray(posterior_pr90, dims=pr90_truth.dims, coords=pr90_truth.coords)
    posterior_pr90_std_data = xr.DataArray(posterior_pr90_stddev, dims=pr90_truth.dims, coords=pr90_truth.coords)

    # Save prediction
    posterior_pr90_data.to_netcdf('./outputs/GP-posterior-mean-pr90-test-2019-2100.nc')
    posterior_pr90_std_data.to_netcdf('./outputs/GP-posterior-std-pr90-test-2019-2100.nc')


def emulate_dtr():
    # Get data
    Xtrain, eof_solvers = create_predictor_data(train_files)
    Ytrain_dtr = create_predictdand_data(train_files)['diurnal_temperature_range'].values.reshape(-1, 96*144)
    Xtest = get_test_data('ssp245', eof_solvers)
    Ytest = xr.open_dataset('./test/outputs_ssp245.nc').compute()
    dtr_truth = Ytest['diurnal_temperature_range'].mean('member')
    
    # Drop rows including nans
    train_nan_mask = Xtrain.isna().any(axis=1).values
    Xtrain = Xtrain.dropna(axis=0, how='any')
    Ytrain_dtr = Ytrain_dtr[~train_nan_mask]
    assert Xtrain.shape[0]==Ytrain_dtr.shape[0]
    
    test_nan_mask = Xtest.isna().any(axis=1).values
    Xtest = Xtest.dropna(axis=0, how='any')
    dtr_truth = dtr_truth[~test_nan_mask]
    
    # Standardize predictor fields requiring standardization (non-EOFs)
    train_CO2_mean, train_CO2_std = Xtrain['CO2'].mean(), Xtrain['CO2'].std()
    train_CH4_mean, train_CH4_std = Xtrain['CH4'].mean(), Xtrain['CH4'].std()
    
    Xtrain['CO2'] = (Xtrain['CO2'] - train_CO2_mean) / train_CO2_std
    Xtrain['CH4'] = (Xtrain['CH4'] - train_CH4_mean) / train_CH4_std
    
    Xtest['CO2'] = (Xtest['CO2'] - train_CO2_mean) / train_CO2_std
    Xtest['CH4'] = (Xtest['CH4'] - train_CH4_mean) / train_CH4_std
    
    # Standardize predictand fields
    train_dtr_mean, train_dtr_std = Ytrain_dtr.mean(), Ytrain_dtr.std()
    Ytrain_dtr = (Ytrain_dtr - train_dtr_mean) / train_dtr_std

    # Define kernels
    kernel_CO2 = gpflow.kernels.Matern32(active_dims=[0]) # active_dims specifies which dimension the kernel is applied to
    kernel_CH4 = gpflow.kernels.Matern32(active_dims=[1])
    kernel_BC = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[2, 3, 4, 5, 6])
    kernel_SO2 = gpflow.kernels.Matern32(lengthscales=5 * [1.], active_dims=[7, 8, 9, 10, 11])
    kernel = kernel_CO2 + kernel_CH4 + kernel_BC + kernel_SO2
    
    # Define Model
    mean = gpflow.mean_functions.Constant()
    
    model = gpflow.models.GPR(data=(Xtrain.astype(np.float64), # cast to float64 because gpflow requires numerical stability
                                    Ytrain_dtr.astype(np.float64)),
                              kernel = kernel,
                              mean_function = mean)
    
    # Define optimizer
    optimizer = gpflow.optimizers.Scipy()
    
    # Train
    optimizer.minimize(model.training_loss,
                       variables=model.trainable_variables,
                       options=dict(disp=True, maxiter=1000))
    
    # Predict
    standard_posterior_mean, standard_posterior_var = model.predict_y(Xtest.values) # predicted mean of GP, predicted variance of GP
    posterior_mean = standard_posterior_mean * train_dtr_std + train_dtr_mean # transform mean prediction to original scale
    posterior_stddev = np.sqrt(standard_posterior_var) * train_dtr_std # transform variance prediction to original scale standard deviation
    
    # Put output back into xarray format for calculating RMSE/plotting
    posterior_dtr = np.reshape(posterior_mean, [86, 96, 144])
    posterior_dtr_stddev = np.reshape(posterior_stddev, [86, 96, 144])
    
    posterior_dtr_data = xr.DataArray(posterior_dtr, dims=dtr_truth.dims, coords=dtr_truth.coords)
    posterior_dtr_std_data = xr.DataArray(posterior_dtr_stddev, dims=dtr_truth.dims, coords=dtr_truth.coords)

    # Save prediction
    posterior_dtr_data.to_netcdf('./outputs/GP-posterior-mean-dtr-test-2019-2100.nc')
    posterior_dtr_std_data.to_netcdf('./outputs/GP-posterior-std-dtr-test-2019-2100.nc')


if __name__ == '__main__':
    emulate_tas()
    print('Prediction for surface air temperature has been successfully saved to the outputs directory')
    emulate_pr()
    print('Prediction for precipitation has been successfully saved to the outputs directory')
    emulate_pr90()
    print('Prediction for 90th percentile precipitation has been successfully saved to the outputs directory')
    emulate_dtr()
    print('Prediction for diurnal temperature range has been successfully saved to the outputs directory')
    