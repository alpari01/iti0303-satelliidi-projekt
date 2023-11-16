#!/bin/bash

pip install tensorflow scikit-learn tifffile pandas psutil matplotlib seaborn tensorflow-gpu

python_executable="./tif_build_model.py"

python3 $python_executable