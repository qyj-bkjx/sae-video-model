#!/bin/bash
# ============================================================
# example accelerate train 8GPU
# ============================================================

set -e

# Environment variables
export PYTHONPATH="$(pwd):$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Training parameters
CONFIG="configs/training/mova_train_accelerate_8gpu.py"
ACCELERATE_CONFIG="configs/training/accelerate/fsdp_8gpu.yaml"

accelerate launch \
        --config_file ${ACCELERATE_CONFIG} \
        scripts/training_scripts/accelerate_train.py ${CONFIG}
