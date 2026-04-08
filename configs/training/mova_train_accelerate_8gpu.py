# ============================================================
# MOVA LoRA Training Configuration
# ============================================================

# --------------------------------------------------
# Model Configuration
# --------------------------------------------------
diffusion_pipeline = dict(
    type="MOVATrain_from_pretrained",
    from_pretrained="/path/to/pretrained_model",
    use_gradient_checkpointing=True,
    use_gradient_checkpointing_offload=True,
)
# --------------------------------------------------
# Data Configuration
# --------------------------------------------------
data = dict(
    dataset=dict(
        type="VideoAudioDataset",
        data_root="/path/to/data",
        metadata_file="metadata.json",
        num_frames=193,  
        height=352,     
        width=640,     
        sample_rate=48000,
        video_fps=24.0,
    ),
    transform=None,
    batch_size=1,  
    num_workers=4,
)

# --------------------------------------------------
# Optimizer Configuration
# --------------------------------------------------
optimizer = dict(
    type="AdamW",
    lr=1e-4,  
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8,
)

# --------------------------------------------------
# LoRA Configuration
# --------------------------------------------------
lora = dict(
    rank=16,              
    alpha=16.0,           
    dropout=0.0,         
    target_modules=[    
        "q", "k", "v", "o",       
        "to_q", "to_k", "to_v",  
        "proj",                    
    ],
)

# --------------------------------------------------
# FSDP Configuration (Optional)
# --------------------------------------------------
fsdp = dict(
    sharding_strategy="FULL_SHARD",  # "FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"
    cpu_offload=True, 
    backward_prefetch="BACKWARD_PRE", 
    reshard_after_forward=True, 
)

# --------------------------------------------------
# Logger Configuration
# --------------------------------------------------
# TensorBoard configuration (Offline environment recommended)
logger = dict(
    log_dir="./tensorboard/muli_gpu", 
)

# WandB configuration (Online environment recommended)
# logger = dict(
#     project="mova-lora-training",
#     name=None,  # Automatically generated
#     tags=["lora", "video-audio"],
#     dir="./wandb",
# )

# --------------------------------------------------
# Trainer Configuration
# --------------------------------------------------
trainer = dict(
    # Training steps
    max_steps=50000,
    num_train_timesteps=1000,
    
    # Gradient
    gradient_accumulation_steps=4, 
    gradient_clip_norm=1.0,
    
    # Mixed precision
    mixed_precision="bf16",  
    
    # FSDP
    use_fsdp=True, 
    
    # Warmup
    warmup_steps=500,
    lr_scheduler_type="cosine",
    min_lr=1e-6,
    
    # Logging
    log_interval=1,
    logger_type="tensorboard",  # "wandb", "tensorboard", "both", "none" (Offline environment uses tensorboard)
    
    # Checkpointing
    save_interval=200,
    save_path="./checkpoints/multi/mova_lora",
    resume_from=None,
    
    train_modules=["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"],
    
    # LoRA
    use_lora=True,

    # Context Parallel
    enable_cp=True,
)