# Landmask Indexing and Classification


## Description
This repository contains a Python implementation of an algorithm designed for landmask indexing. The algorithm processes satellite imagery in GeoTIFF format and identifies specific regions of interest based on predefined criteria. The identified regions are then classified into different classes using a machine learning model trained on relevant datasets.## Table of Contents
- [Components](#components)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Customization](#customization)

## Components
### 1. Machine Learning Module (`models/tif_model.py`):
- Implements a machine learning model (Random Forest in this case) to classify image regions.
- Reads and preprocesses datasets containing labeled examples for training.
- Trains the model and evaluates its performance on test data.
- Provides functionality for predicting classes for new data.

### 2. Indexing Module (`/landmask_indexing/landmask_indexing.py`):
- Reads and processes GeoTIFF satellite imagery.
- Applies a predefined algorithm to identify regions of interest (landmask) within the images.
- Utilizes a machine learning model to classify the identified regions into different classes.
- Generates an output plot overlaying the original landmask with the identified classes.

## Getting Started
1. Clone the repository `git clone https://github.com/alpari01/iti0303-satelliidi-projekt.git`

2. Install dependencies `pip install numpy matplotlib seaborn pandas tifffile scikit-learn`

### GDAL installation
1) Go [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal) and download `.whl` file.

2) Install it `pip install /path/to/downloaded_file.whl`

## Usage Examples
### 1. Machine Learning Module
```
model.tif_images_path = "/illukas/data/projects/iti_wave_2023/tif_images"
model.measurements_path = "/illukas/data/projects/iti_wave_2023/measurements"
model.pickle_path = "/illukas/data/projects/iti_wave_2023/iti0303-satelliidi-projekt/datasets"
model.build_dataset(64, 40, 1200)
model.model_build_rf()
model.model_fit(64)
```
### 2. Indexing Module
```
tif_file = r'landmask_indexing/tiff/S1A_IW_GRDH_1SDV_20221027T160500_20221027T160525_045630_0574C2_211B_Cal_Spk_TC.tif'
shapefile = r'landmask_indexing/tiff/gshhg-shp-2.3.7/GSHHS_shp/f/GSHHS_f_L1.shp'
datasets_folder = r'/models/datasets/'
```


## Customization

In `landmask_indexing.py` and `grid_plot.py`, you can customize the following parameters:
  - `max_square`: Set the maximum square size for indexing.
  - `min_square`: Set the minimum square size for indexing.
  - `filling`: Control the filling criteria for indexing the landmask.

## Our suggestions on model improvements
- Problem with bigger images. On bigger satellite images there are not squares of one size, there are squares of sizes from 32 to 512px. We assume, that when the model sees, a data of .tif square, for example of size 512, then it doesn’t actually know what size it is (since it only sees its calculated std, mean, percentile_25 and percentile_75 values). So when we ask the model to predict the result based on all the squares together 32, 64, 128, 256 and 512, then for 512 squares it will predict good results on a dataset of 512, but for others squares that are not sized 512 it will provide poor results.
- Possible solutions
  - Try to add square size parameter to features set, so it is like so **std, mean, percentile_25, percentile_75, square_size**.
