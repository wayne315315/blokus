#!/bin/bash

# git clone
git clone https://github.com/wayne315315/blokus.git
git checkout torch
cd blokus

# Install dependencies
apt update && apt install -y python3.12-dev python3.12-venv
python3.12 -m venv .venv
printf '\n# Manually set CUDA 13.0 paths for Verda B300\nexport CUDA_HOME=/usr/local/cuda-13.0\nexport LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH\n' >> .venv/bin/activate
source .venv/bin/activate

# install pip packages
pip install -r requirements.txt
python setup.py build_ext --inplace
