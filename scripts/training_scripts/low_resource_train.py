#!/usr/bin/env python3
"""
MOVA Low Resource LoRA Training Script

This script implements memory-efficient training using:
1. LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
2. Gradient checkpointing to reduce activation memory
3. Optional 8-bit optimizer to reduce optimizer state memory
4. Optional FP8 CPU offloading for frozen weights
```
"""

import sys
import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from mmengine.config import Config, DictAction

# Local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from mova.registry import DATASETS, DIFFUSION_PIPELINES, TRANSFORMS
from mova.datasets.video_audio_dataset import collate_fn
from mova.engine.trainer.low_resource.low_resource_trainer import LowResourceTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="MOVA Low Resource Training")
    parser.add_argument("config", help="Config file path")
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options'
    )
    
    return parser.parse_args()




def build_dataloader(cfg):
    """Build data loader."""
    # Build transforms
    transform = None
    if cfg.get("transform", None) is not None:
        transform = TRANSFORMS.build(cfg.transform)
    
    # Build dataset
    dataset_cfg = cfg.dataset.copy()
    dataset_cfg["transform"] = transform
    dataset = DATASETS.build(dataset_cfg)
    
    # Build DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    
    return dataloader


def main():
    args = parse_args()
    
    # Load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    print(f"Config: {args.config}")
    
    # Set device (read local_rank from env for distributed training)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    print(f"Device: cuda:{local_rank if local_rank >= 0 else 0}")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    
    # Build dataloader
    print("\nBuilding dataloader...")
    dataloader = build_dataloader(cfg.data)
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # Build model
    print("\nBuilding model...")
    model = DIFFUSION_PIPELINES.build(
        cfg.diffusion_pipeline,
        default_args={"device": "cpu", "torch_dtype": torch.bfloat16}  # Load to CPU first
    )
    
    # Setup scheduler for training
    model.scheduler.set_timesteps(
        cfg.trainer.get("num_train_timesteps", 1000),
        training=True
    )
    
    # Create trainer
    trainer = LowResourceTrainer(
        model=model,
        train_dataloader=dataloader,
        cfg=cfg,
    )
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
