#!/bin/bash
set -e

# Run model training (including LOGO cross-validation)
python src/model_training.py

# Generate and save visualization results
python src/visualize.py
