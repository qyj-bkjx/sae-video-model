#!/bin/bash
# ============================================================
# example accelerate train 1GPU
# ============================================================

set -e

# Environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

# Training parameters
CONFIG="configs/training/mova_train_accelerate.py"

python scripts/training_scripts/accelerate_train.py ${CONFIG}