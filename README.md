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
3. Ensure you have GDAL installed on your system. You can install it using system-specific package managers or download it from the official GDAL website.


## Usage
### `read_geotiff.py`
The `read_geotiff.py` script provides functions for working with GeoTIFF files. It includes the following functions:
- `get_landmask_and_error()`: Extracts a landmask from a GeoTIFF file based on a provided shapefile and returns it.
- `get_raster_band()`: Reads a raster band from a GeoTIFF file and returns it as a NumPy array.

### `landmask_indexing.py`
The `landmask_indexing.py` script is designed for landmask indexing. It includes the following function:
- `get_indices()`: Generates indices for landmask squares of a specified size and fills a list with square coordinates. The result is a list of top-left and bottom-right coordinates for landmask squares.

### `grid_plot.py`
The `grid_plot.py` script is responsible for creating visual representations of the landmask squares on top of GeoTIFF images. It includes the following functions:
- `draw_grid()`: Draws landmask squares on the input data, providing a visual representation of the indexed squares.
- `transfer_indices()`: Transfers indices from the landmask to a target dataset, allowing for visual representation and analysis.

## Customization

In `landmask_indexing.py` and `grid_plot.py`, you can customize the following parameters:
  - `max_square`: Set the maximum square size for indexing.
  - `min_square`: Set the minimum square size for indexing.
  - `filling`: Control the filling criteria for indexing the landmask.