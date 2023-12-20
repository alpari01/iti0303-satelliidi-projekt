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

## Usage
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

In `landmask_indexing.py` you can customize the following parameters:
  - `max_square`: Set the maximum square size for indexing.
  - `min_square`: Set the minimum square size for indexing.
  - `filling`: Control the filling criteria for indexing the landmask.