#!/bin/bash
# ============================================================
# MOVA Low Resource LoRA Training Script
# This script runs memory-efficient training using:
# - LoRA for parameter-efficient fine-tuning
# - Gradient checkpointing to reduce activation memory
# - Optional 8-bit optimizer (bitsandbytes)
# - Optional FP8 CPU offloading for frozen weights
# ============================================================

# Exit on error
set -e

# Configuration
CONFIG_FILE=${1:-"configs/training/mova_train_low_resource.py"}

# Environment setup
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Memory optimization settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export DEBUG_OUTPUT=0

echo "============================================================"
echo "MOVA Low Resource LoRA Training"
echo "============================================================"
echo "Config: ${CONFIG_FILE}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Note: All training parameters are in the config file"
echo "============================================================"

# Check if bitsandbytes is installed
python -c "import bitsandbytes" 2>/dev/null || {
    echo "[Warning] bitsandbytes not installed, using standard AdamW"
    echo "Install with: pip install bitsandbytes"
    echo "This will use more GPU memory for optimizer states."
}

# Build command
# All training parameters are in the config file, here only pass the config file path
CMD="python scripts/training_scripts/low_resource_train.py \
    ${CONFIG_FILE}"

echo ""
echo "Starting training (with FP8 CPU offload, most memory efficient)..."
# Run training
eval ${CMD}

echo ""
echo "============================================================"
echo "Training completed! Check checkpoint save_path in config file"
echo "============================================================"
