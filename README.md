# DSC-180A Capstone Project - Section B13

**Note:** Access to NCAR is required to execute the code in the NCAR directory.

## Project Overview

This project aims to replicate the results from the paper *ClimateBench v1.0: A Benchmark for Data-Driven Climate Projections*. The goal is to develop an alternative solution for climate modeling that bypasses the computational expense of Earth System Models. [ClimateBench Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002954)



## Running the Models

To run the models:

1. Download `train_val.tar.gz` and `test.tar.gz` from [10.5281/zenodo.5196512](https://zenodo.org/records/7064308).
2. Extract the files into the `train_val` and `test` folders, respectively.
3. Alternatively, you can modify the file paths in `utils.py` to match your data locations.
4. Recreate the conda environment using ```conda env create -f environment.yml```


## Notebooks

The `notebooks` folder contains all the Jupyter notebooks used for model development.


