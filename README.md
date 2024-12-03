# DSC-180A Capstone Project - Section B13

**Note:** Access to NCAR is required to execute the code in the NCAR directory.

## Project Overview

This project aims to replicate the results from the paper *ClimateBench v1.0: A Benchmark for Data-Driven Climate Projections*. The goal is to develop an alternative solution for climate modeling that bypasses the computational expense of Earth System Models. [ClimateBench Paper](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021MS002954)



## Running the Models

To run the models, follow these steps:

1. **Download the Data**  
   - Download the files `train_val.tar.gz` and `test.tar.gz` from [10.5281/zenodo.5196512](https://zenodo.org/records/7064308).

2. **Extract the Files**  
   - Extract the contents of `train_val.tar.gz` into a folder named `train_val`.
   - Extract the contents of `test.tar.gz` into a folder named `test`.
   - Place these two folders into `DSC-180A-Capstone-Section-B13`

3. **Modify File Paths (Optional)**  
   - If you prefer to store the data in a different location, update the file paths in `utils.py` to match your data directory structure.

4. **Set Up the Environment**  
   Run the following commands to create and configure a Conda environment:

   ```bash
   conda create -n B13 python=3.10
   conda activate B13
   conda install -c conda-forge notebook xarray matplotlib cartopy eofs scikit-learn
   pip install "esem[gpflow,keras]" netcdf4
   ```

    If you encounter issues with `tensorflow-probability` on macOS, downgrade it to a version compatible with your TensorFlow installation:

    ```bash
    pip install tensorflow-probability==0.24
    ```

    Once the environment is set up, you should be able to proceed with running the models.

