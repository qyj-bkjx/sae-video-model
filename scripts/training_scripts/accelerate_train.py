#!/usr/bin/env python3
"""
MOVA Training Script (Accelerate + FSDP + LoRA)
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import DataLoader
from mmengine.config import Config, DictAction

from mova.registry import DATASETS, DIFFUSION_PIPELINES, TRANSFORMS
from mova.engine.trainer.accelerate.accelerate_trainer import AccelerateTrainer
from mova.datasets.video_audio_dataset import collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="MOVA Training with Accelerate")
    parser.add_argument("config", help="Config file path")
    
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options'
    )
    
    return parser.parse_args()


def build_dataloader(cfg):
    """Build data loader"""
    transform = None
    if cfg.get("transform", None) is not None:
        transform = TRANSFORMS.build(cfg.transform)
    
    dataset_cfg = cfg.dataset.copy()
    dataset_cfg["transform"] = transform
    dataset = DATASETS.build(dataset_cfg)
    
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
    
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    dataloader = build_dataloader(cfg.data)
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Batch size: {cfg.data.batch_size}")
    
    use_fsdp = cfg.trainer.get("use_fsdp", False)
    if use_fsdp:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)  # Set the GPU used by the current process
        
        device = "cpu"
        
        print(f"[FSDP] Process LOCAL_RANK={local_rank}: Loading model on CPU, FSDP will handle device placement and sharding")
    else:
        device = "cpu"
    
    torch_dtype = torch.bfloat16
    # pass pipeline-level kwargs (include gradient checkpointing flags if set)
    model_default_args = {"device": device, "torch_dtype": torch_dtype}
    if getattr(cfg, 'diffusion_pipeline', None) is not None:
        if cfg.diffusion_pipeline.get('use_gradient_checkpointing', False):
            model_default_args['use_gradient_checkpointing'] = True
        if cfg.diffusion_pipeline.get('use_gradient_checkpointing_offload', False):
            model_default_args['use_gradient_checkpointing_offload'] = True

    model = DIFFUSION_PIPELINES.build(
        cfg.diffusion_pipeline,
        default_args=model_default_args,
    )
    
    model.scheduler.set_timesteps(
        cfg.trainer.get("num_train_timesteps", 1000),
        training=True
    )
    

    
    logger_kwargs = {}
    if hasattr(cfg, 'logger'):
        logger_kwargs = dict(cfg.logger)
    
    trainer = AccelerateTrainer(
        model=model,
        train_dataloader=dataloader,
        optimizer_cfg=dict(cfg.optimizer),
        # Training
        max_steps=cfg.trainer.max_steps,
        gradient_accumulation_steps=cfg.trainer.get("gradient_accumulation_steps", 1),
        gradient_clip_norm=cfg.trainer.get("gradient_clip_norm", 1.0),
        # Mixed precision
        mixed_precision=cfg.trainer.get("mixed_precision", "bf16"),
        # FSDP
        use_fsdp=cfg.trainer.get("use_fsdp", False),
        fsdp_config=dict(cfg.fsdp) if hasattr(cfg, 'fsdp') else None,
        # Logging
        log_interval=cfg.trainer.get("log_interval", 10),
        logger_type=cfg.trainer.get("logger_type", "wandb"),
        logger_kwargs=logger_kwargs,
        # Checkpointing
        save_interval=cfg.trainer.get("save_interval", 1000),
        save_path=cfg.trainer.get("save_path", "./checkpoints"),
        resume_from=cfg.trainer.get("resume_from", None),
        # LR Scheduler
        lr_scheduler_type=cfg.trainer.get("lr_scheduler_type", "cosine"),
        warmup_steps=cfg.trainer.get("warmup_steps", 1000),
        min_lr=cfg.trainer.get("min_lr", 1e-6),
        # Training modules
        train_modules=cfg.trainer.get("train_modules", None),
        # LoRA
        use_lora=cfg.trainer.get("use_lora", False),
        lora_config=dict(cfg.lora) if hasattr(cfg, 'lora') else None,
        # Context Parallel is handled by Accelerate's parallelism_config
        enable_cp=cfg.trainer.get("enable_cp", False),
    )
    
    print("Starting training...")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()

