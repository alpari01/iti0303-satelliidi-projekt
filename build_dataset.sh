#!/bin/bash

pip install tensorflow scikit-learn tifffile pandas psutil

python_executable="./tif_build_dataset.py"

python3 $python_executable