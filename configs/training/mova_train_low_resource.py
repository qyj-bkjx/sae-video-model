# ============================================================
# MOVA LoRA Training Configuration with FP8 CPU Offload
# 
# This config enables ultra-memory-efficient training using:
# 1. FP8 quantization of frozen weights stored on CPU
# 2. weight loading/offloading during forward/backward
# 3. LoRA for parameter-efficient fine-tuning
# 4. Gradient checkpointing with CPU offload
# 5. AdamW 8-bit optimizer
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
# Optimizer Configuration (8-bit AdamW)
# --------------------------------------------------
optimizer = dict(
    type="AdamW8bit",  
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    eps=1e-8,
)

# --------------------------------------------------
# LoRA Configuration
# --------------------------------------------------
lora = dict(
    rank=8,               
    alpha=8.0,            
    dropout=0.0,          
    target_modules=[      
        "q", "k", "v", "o",
        "to_q", "to_k", "to_v", "to_out",
        "proj",
    ],
)

# --------------------------------------------------
# FP8 CPU Offload Configuration
# --------------------------------------------------
fp8_offload = dict(
    enabled=True,
    # Modules to FP8-offload
    target_modules=["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"],
    # Excluded parameter patterns (LoRA params are not offloaded)
    exclude_patterns=["lora_A", "lora_B", "lora_"],
)


# --------------------------------------------------
# Logger Configuration
# --------------------------------------------------
logger = dict(
    log_dir="./tensorboard/mova_fp8_lora",
)

# --------------------------------------------------
# Trainer Configuration
# --------------------------------------------------
trainer = dict(
    # Training steps
    max_steps=50000,
    num_train_timesteps=1000,
    
    # Gradient settings
    gradient_accumulation_steps=4,  
    gradient_clip_norm=1.0,
    
    # Mixed precision
    mixed_precision="bf16",
    
    # FP8 CPU Offload 
    use_fp8_cpu_offload=True,
    
    # Gradient checkpointing 
    gradient_checkpointing=True,
    gradient_checkpointing_offload=True,  
    
    # Warmup and LR scheduler
    warmup_steps=500,
    lr_scheduler_type="cosine",
    min_lr=1e-6,
    
    # Logging
    log_interval=1,
    logger_type="tensorboard",
    
    # Checkpointing
    save_interval=100,
    save_path="./checkpoints/mova_lora_low_resource", 
    resume_from=None,
    

    train_modules=["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"],
    
    # LoRA
    use_lora=True,
)