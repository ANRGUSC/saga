#!/bin/bash
cd "$(dirname "$0")"

# install SAGA
git checkout fb4a4e5f5070053e3902516638e6ce41356ac51c
pip install -e ..

# Install experiment requirements
pip install -r ./requirements.txt

# # Download the data
wget https://zenodo.org/records/10901274/files/dataset.zip?download=1 -O dataset.zip
# unzip into ./datasets/parametric_benchmarking - dataset.zip contains the directory parametric_benchmarking
mkdir -p ./datasets
unzip dataset.zip -d ./datasets
rm dataset.zip

# Run the experiments
python exp_parametric.py run \
    --datadir "./datasets/parametric_benchmarking" \
    --out "./results/parametric.csv" \
    --trim 100 --batch -1 --batches -1

python post_parametric.py
