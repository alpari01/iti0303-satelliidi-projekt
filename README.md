# Landmask Indexing Algorithm


## Description
This repository contains a set of Python scripts for geospatial data processing, specifically designed for working with GeoTIFF files. The code allows you to analyze, manipulate, and visualize geospatial data with a focus on landmask extraction, indexing, and grid plotting.


## Table of Contents
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Customization](#customization)


## Prerequisites
- [Python](https://www.python.org/): This project is written in Python 3.7+.
- [GDAL](https://gdal.org/): The Geospatial Data Abstraction Library is used to handle GeoTIFF files.
- [Matplotlib](https://matplotlib.org/): This library is used for creating visualizations.
- [Seaborn](https://seaborn.pydata.org/): Seaborn complements Matplotlib for creating attractive statistical graphics.
- [NumPy](https://numpy.org/): NumPy is used for efficient numerical operations.


## Getting Started
1. Clone the repository to your local machine. 
```
git clone https://github.com/alpari01/iti0303-satelliidi-projekt.git
```
2. Install the required dependencies. You can use pip to install the necessary packages.
```
pip install gdal numpy matplotlib seaborn
```

### Troubleshooting GDAL installation
1) Go to https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal and download `.whl` file that is compatible with your operating system and python version.

2) Use this command to install directly from `.whl` file instead.
```
pip install /path/to/downloaded_file.whl
```

## Usage
### Build dataset
**Create a new model object**
```
model = TifModel()
```
**Configure .tif images path.** 
```
model.tif_images_root_path = "/your/path/to/tif/images"
```
This path should contain all your .tif images and must have the following structure:
- **root**
  - **gof_gcp_2**
    - *tif_image*
    - *tif_image*
    - ...
    - *tif_image*
  - **knolls_gcp_2**
    - ... 
  - **nbp_gcp_2**
    - ... 
  - **selka_gcp_2**
    - ...  

**Configure measurements root path.** 
```
model.measurements_root_path = "/your/path/to/tif/measurements"
```
This path should contain measurements for your .tif images and must have the following structure:
- **root**
  - **format_gof.csv**
  - **format_knolls.csv**
  - **format_nbp.csv**
  - **format_selka.csv**
 
**Configure pickle path.** 
```
model.pickle_path = "/your/path/to/save/pickle/file/to"
```
This is path where dataset pickle file will be saved to.

**Build dataset**
```
model.build_dataset(64, 40, 1200)
```
This method will read all images from _tif_images_root_path_, crop them to size _64x64_ pixels, discarding any images below _40MB_ and save built dataset to _pickle_path_.

This dataset will have features **std, mean, percentile_25, percentile_75** and labels **HS class**.

HS classes definitions:
- 0: hs <= 0.5
- 1: 0.5 < hs <= 1.0
- 2: 1.0 < hs <= 1.5
- 3: 1.5 < hs <= 2.0
- 4: 2.0 < hs <= 2.5
- 5: 2.5 < hs

This dataset will be balanced and have ~200 images per each HS class.

You will see created dataset pickle file with name _data-64px.pkl_ appear in your _pickle_path_ directory.

### Build and fit model
We found that with the current solution RandomForest model provides the most accurate results.

You can build a RandomForest model like that
```
model.model_build_rf()
```

Then you can fit the model. This methdod will automatically look up for pickle file with name _data-64px.pkl_ and use it for model fitting.
```
model.model_fit(64)
```
Model confusion matrix will be created in the same directory you launch script from and will have name _confusion-matrix-64px.png_.


## Customization

In `landmask_indexing.py` and `grid_plot.py`, you can customize the following parameters:
  - `max_square`: Set the maximum square size for indexing.
  - `min_square`: Set the minimum square size for indexing.
  - `filling`: Control the filling criteria for indexing the landmask.

## Our suggestions on model improvements
- Problem with bigger images. On bigger satellite images there are not squares of one size, there are squares of sizes from 32 to 512px. We assume, that when the model sees, a data of .tif square, for example of size 512, then it doesn’t actually know what size it is (since it only sees its calculated std, mean, percentile_25 and percentile_75 values). So when we ask the model to predict the result based on all the squares together 32, 64, 128, 256 and 512, then for 512 it will predict good results on a dataset of 512, but for others it will provide poor results.
- Possible solutions
  - Try to add square size parameter to features set, so it is like so **std, mean, percentile_25, percentile_75, square_size**.
