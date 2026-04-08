#!/usr/bin/env bash

# Model trained with Accelerate Trainer
export ACCELERATE_TRAINER_TRAINED=1
# # Model trained with Low Resource Trainer
# export ACCELERATE_TRAINER_TRAINED=0


torchrun --nproc_per_node=8 scripts/inference_single_lora.py \
    --base_model /path/to/pretrained_model \
    --lora_path /path/to/lora_model \
    --prompt "Wearing professional attire and a white hard hat, a woman stands before a modern glass-fronted building, a roll of blueprints tucked in her arm. Footsteps echo softly around her as she faces the camera and explains, \"The main structure of the building has been completed, but the plumbing and electrical systems in many rooms are still being installed. It is expected to be operational in a month.\"" \
    --ref_path "./assets/single_person.jpg" \
    --output_path "./data/samples/single_person.mp4" \
    --cp_size 8