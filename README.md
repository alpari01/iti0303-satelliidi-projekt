# Landmask Indexing Algorithm


## Description
This repository contains a Python script for a landmask indexing algorithm that processes a CSV file containing binary data (0s and 1s) to identify regions with a specified fill ratio of 1s. The algorithm recursively subdivides the input data into smaller squares until it finds regions that meet the specified criteria.


## Table of Contents
- [Prerequisites](#prerequisites)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Algorithm Overview](#algorithm-overview)


## Prerequisites
- Python 3.x
- `csv` library


## Getting Started
1. Clone this repository to your local machine or download the `landmask_indexing.py` file.
2. Ensure you have the required prerequisites installed.
3. Prepare your landmask data in a CSV file. The landmask should consist of binary values (0 for land and 1 for non-land) in a grid format.


## Usage
1. Modify the `landmask_filename` variable in the `landmask_indexing.py` script to specify the path to your CSV file containing the landmask data.
2. Run the script by executing the following command in your terminal:
3. The script will process the landmask data, and it will print out the execution time and the identified indices (regions) that meet the specified fill ratio criteria.


## Algorithm Overview

The `get_indices` function in the script implements a recursive algorithm to find regions in the landmask data with a specified fill ratio of 1s. Here's an overview of the algorithm:

1. Read the CSV file containing the landmask data.
2. Initialize the starting parameters for the algorithm, including the maximum square size (`max_square`), minimum square size (`min_square`), and filling ratio (`filling`).
3. The algorithm starts with the entire landmask data and iteratively subdivides it into smaller squares until it finds regions that contain a proportion of 1s greater than or equal to the specified filling ratio.
4. The identified regions are stored in the `indices` list as pairs of top-left and bottom-right coordinates.
