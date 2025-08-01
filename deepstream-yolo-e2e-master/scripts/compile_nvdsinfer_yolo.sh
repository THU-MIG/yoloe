#!/bin/bash

# Creative Commons Attribution-NonCommercial 4.0 International License
# 
# You are free to share and adapt the material under the following terms:
# - Attribution: Give appropriate credit.
# - NonCommercial: Not for commercial use without permission.
# 
# For inquiries: levi.pereira@gmail.com
# Repository: DeepStream / YOLO (https://github.com/levipereira/deepstream-yolo-e2e)
# License: https://creativecommons.org/licenses/by-nc/4.0/legalcode

# Detect installed CUDA version
cuda_dir=$(ls -d /usr/local/cuda-*/ 2>/dev/null | grep -oP 'cuda-\K[0-9]+\.[0-9]+')

if [ -z "$cuda_dir" ]; then
    echo "No CUDA installation found in /usr/local/"
    exit 1
else
    echo "Detected CUDA version: $cuda_dir"
fi

# Set the CUDA_VER environment variable
export CUDA_VER=$cuda_dir

# Execute the make command with the detected CUDA version
make install -C nvdsinfer_yolo CUDA_VER=$CUDA_VER   

