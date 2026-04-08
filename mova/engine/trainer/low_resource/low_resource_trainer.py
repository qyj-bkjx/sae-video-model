"""
Low Resource Trainer for MOVA LoRA Training

This trainer implements memory-efficient training using:
1. LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning
2. Gradient checkpointing to reduce activation memory
3. Optional 8-bit optimizer to reduce optimizer state memory
4. Optional FP8 CPU offloading for frozen weights
"""

import os
import re
import gc
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from mmengine.config import Config
from tqdm import tqdm

from mova.engine.trainer.low_resource.lora_layers import LoRAManager
from mova.engine.trainer.utils.logger import build_logger

# 8-bit optimizer
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    print("[Warning] DeepSpeed not available, will use standard PyTorch training")

def get_8bit_optimizer(params, lr, betas, weight_decay, eps):
    """Create an 8-bit AdamW optimizer."""
    if BNB_AVAILABLE:
        optimizer = bnb.optim.AdamW8bit(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
        print("[Optimizer] Using AdamW 8-bit (bitsandbytes)")
    elif DEEPSPEED_AVAILABLE:
        optimizer = DeepSpeedCPUAdam(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
        print("[Optimizer] Using DeepSpeedCPUAdam")
    else:
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            eps=eps,
        )
        print("[Optimizer] Using standard AdamW")
    
    return optimizer


def create_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr=1e-6):
    """Create a cosine LR scheduler with warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (max_steps - warmup_steps)
            return max(min_lr / optimizer.defaults['lr'], 
                      0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item()))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class LowResourceTrainer:
    """
    Low resource trainer with LoRA support.
    
    This trainer implements:
    1. LoRA training with gradient checkpointing
    2. Optional 8-bit optimizer for reduced memory
    3. Optional FP8 CPU offloading for frozen weights
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        cfg: Config,
    ):
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        
        # Training config
        trainer_cfg = cfg.trainer
        self.max_steps = trainer_cfg.max_steps
        self.gradient_accumulation_steps = trainer_cfg.get("gradient_accumulation_steps", 1)
        self.gradient_clip_norm = trainer_cfg.get("gradient_clip_norm", 1.0)
        self.warmup_steps = trainer_cfg.get("warmup_steps", 1000)
        self.log_interval = trainer_cfg.get("log_interval", 10)
        self.logger_type = trainer_cfg.get("logger_type", "tensorboard")
        self.save_interval = trainer_cfg.get("save_interval", 1000)
        self.save_path = trainer_cfg.get("save_path", "./checkpoints/mova_lora")
        
        # Logger config
        logger_cfg = cfg.get("logger", {})
        self.logger_kwargs = dict(logger_cfg)
        
        # LoRA config
        lora_cfg = cfg.get("lora", {})
        self.lora_rank = lora_cfg.get("rank", 8)
        self.lora_alpha = lora_cfg.get("alpha", 8.0)
        self.lora_dropout = lora_cfg.get("dropout", 0.0)
        self.lora_target_modules = lora_cfg.get("target_modules", [
            "q", "k", "v", "o", "proj", "to_q", "to_k", "to_v", "to_out"
        ])
        
        # Training modules
        self.train_modules = trainer_cfg.get("train_modules", [
            "video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"
        ])
        
        # Dataloader
        self.train_dataloader = train_dataloader
        
        # Initialize model with LoRA
        self.model = self._setup_model(model)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup LR scheduler
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer, 
            self.warmup_steps, 
            self.max_steps,
            trainer_cfg.get("min_lr", 1e-6)
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        # Setup logger
        self.logger = build_logger(
            logger_type=self.logger_type,
            is_main_process=True,  # Low resource trainer is single GPU
            **self.logger_kwargs
        )
        
        # Resume if specified
        resume_from = trainer_cfg.get("resume_from", None)
        if resume_from is None:
            resume_from = self._find_latest_checkpoint()
        if resume_from:
            self._load_checkpoint(resume_from)

    def _find_latest_checkpoint(self):
        """
        Find the latest step-* checkpoint under self.save_path.
        Returns: checkpoint path or None
        """
        if not os.path.isdir(self.save_path):
            return None

        step_pattern = re.compile(r"^step-(\d+)$")

        candidates = []

        for name in os.listdir(self.save_path):
            match = step_pattern.match(name)
            if match:
                step = int(match.group(1))
                full_path = os.path.join(self.save_path, name)
                if os.path.isdir(full_path):
                    candidates.append((step, full_path))

        if not candidates:
            return None

        # Sort by step and take the largest.
        candidates.sort(key=lambda x: x[0])
        return candidates[-1][1]
    
    def _setup_model(self, model: nn.Module) -> nn.Module:
        """Setup model with LoRA injection and optional FP8 offloading."""
        print("\n" + "="*60)
        print("Setting up model with LoRA...")
        print("="*60)
        
        # Move prompter components if needed
        if hasattr(model, 'prompter') and model.prompter is not None:
            if hasattr(model.prompter, 'text_encoder'):
                model.prompter.text_encoder = model.text_encoder  # Share reference
        
        # Freeze all parameters and set to eval
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        # Inject LoRA into target modules
        print(f"\n[LoRA] Injecting LoRA (rank={self.lora_rank}, alpha={self.lora_alpha})")
        print(f"[LoRA] Target modules: {self.lora_target_modules}")
        print(f"[LoRA] Train modules: {self.train_modules}")

        self.lora_manager = LoRAManager(
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
            target_modules=self.lora_target_modules,
            dtype=self.dtype,
        )
        
        model = self.lora_manager.inject_lora(model, self.train_modules)


        # Optional FP8 CPU offloading for frozen weights
        if self.cfg.trainer.get("use_fp8_cpu_offload", False):
            print("\n[FP8 Offload] Setting up FP8 CPU offloading...")
            
            # Exclude LoRA parameters from offloading
            exclude_patterns = self.cfg.get("fp8_offload", {}).get("exclude_patterns", ["lora_A", "lora_B", "lora_"])
            
            # Use the model's built-in FP8 offload setup
            model.setup_fp8_cpu_offload(
                device=self.device,
                target_modules=self.train_modules,
                exclude_patterns=exclude_patterns,
            )
        
        # Enable gradient checkpointing
        if self.cfg.trainer.get("gradient_checkpointing", True):
            print("\n[Gradient Checkpointing] Enabling gradient checkpointing")
            if hasattr(model, 'use_gradient_checkpointing'):
                model.use_gradient_checkpointing = True
                # Default to True for low resource training to prevent memory leak
                model.use_gradient_checkpointing_offload = self.cfg.trainer.get(
                    "gradient_checkpointing_offload", True
                )
                if model.use_gradient_checkpointing_offload:
                    print("[Gradient Checkpointing] CPU offload enabled (activations will be offloaded to CPU)")
        
        # Move LoRA parameters to GPU
        for name, param in model.named_parameters():
            if param.requires_grad and 'lora_' in name:
                param.data = param.data.to(self.device, dtype=self.dtype)
        
        # Print memory usage
        self._print_memory_stats("After model setup")
        
        return model
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        optimizer_cfg = self.cfg.optimizer
        
        # Get trainable parameters (LoRA only)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        print(f"\n[Optimizer] Trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
        
        optimizer = get_8bit_optimizer(
            trainable_params,
            lr=optimizer_cfg.lr,
            betas=optimizer_cfg.get("betas", (0.9, 0.999)),
            weight_decay=optimizer_cfg.get("weight_decay", 0.01),
            eps=optimizer_cfg.get("eps", 1e-8),
        )
        
        return optimizer
    
    def _print_memory_stats(self, stage: str = ""):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            print(f"\n[Memory {stage}]")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved: {reserved:.2f} GB")
            print(f"  Max Allocated: {max_allocated:.2f} GB")
    
    def train_step(self, batch: dict, global_step: int) -> dict:
        """
        Execute a single training step.
        
        Args:
            batch: Dictionary containing training data
            global_step: Current global training step
            
        Returns:
            Dictionary of loss values
        """
        # Move batch to device
        video = batch["video"].to(self.device, dtype=self.dtype)
        audio = batch["audio"].to(self.device, dtype=torch.float32)  # Audio VAE needs fp32
        first_frame = batch["first_frame"].to(self.device, dtype=self.dtype)
        caption = batch["caption"]
        video_fps = batch.get("video_fps", 24.0)
        if isinstance(video_fps, torch.Tensor):
            video_fps = video_fps.item()
            
        # Forward pass with autocast
        with torch.amp.autocast('cuda', dtype=self.dtype):
            self.model.train()
            loss_dict = self.model.training_step(
                video=video,
                audio=audio,
                first_frame=first_frame,
                caption=caption,
                video_fps=video_fps,
                global_step=global_step,
            )
        
        loss = loss_dict["loss"]
        
        # Backward pass
        loss = loss / self.gradient_accumulation_steps
        loss.backward()
        
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in loss_dict.items()}
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)
        print(f"Max steps: {self.max_steps}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.cfg.data.batch_size * self.gradient_accumulation_steps}")
        
        os.makedirs(self.save_path, exist_ok=True)
        
        # Training loop
        data_iter = iter(self.train_dataloader)
        
        accum_loss = 0.0
        accum_video_loss = 0.0
        accum_audio_loss = 0.0
        accum_steps = 0
        
        pbar = tqdm(total=self.max_steps, desc="Training", initial=self.global_step)
        
        while self.global_step < self.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                self.epoch += 1
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            # Training step
            loss_dict = self.train_step(batch, global_step=self.global_step)
            
            accum_loss += loss_dict["loss"]
            accum_video_loss += loss_dict.get("video_loss", 0)
            accum_audio_loss += loss_dict.get("audio_loss", 0)
            accum_steps += 1
            
            # Optimizer step
            if accum_steps >= self.gradient_accumulation_steps:
                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad],
                        self.gradient_clip_norm
                    )
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Clear cache after optimizer step to prevent memory leak
                # For low resource training, we clear cache more frequently
                if self.global_step % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
                # Logging
                if self.global_step % self.log_interval == 0:
                    avg_loss = accum_loss / accum_steps
                    avg_video_loss = accum_video_loss / accum_steps
                    avg_audio_loss = accum_audio_loss / accum_steps
                    lr = self.optimizer.param_groups[0]['lr']
                    
                    # Log to logger (WandB/TensorBoard)
                    metrics = {
                        "train/loss": avg_loss,
                        "train/video_loss": avg_video_loss,
                        "train/audio_loss": avg_audio_loss,
                        "train/lr": lr,
                        "train/epoch": self.epoch,
                    }
                    self.logger.log(metrics, step=self.global_step)
                    
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'v_loss': f'{avg_video_loss:.4f}',
                        'a_loss': f'{avg_audio_loss:.4f}',
                        'lr': f'{lr:.2e}',
                    })
                    
                    # Print memory stats periodically
                    if self.global_step % (self.log_interval * 10) == 0:
                        self._print_memory_stats(f"Step {self.global_step}")
                
                # Save checkpoint
                if self.global_step % self.save_interval == 0:
                    self._save_checkpoint()
                
                # Reset accumulators
                accum_loss = 0.0
                accum_video_loss = 0.0
                accum_audio_loss = 0.0
                accum_steps = 0
                
                pbar.update(1)
        
        pbar.close()
        
        # Final save
        self._save_checkpoint(final=True)
        self.logger.finish()
        print("\n[Training] Complete!")
    
    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        suffix = "final" if final else f"step-{self.global_step}"
        ckpt_dir = os.path.join(self.save_path, suffix)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save LoRA weights
        lora_path = os.path.join(ckpt_dir, "lora_weights.pt")
        self.lora_manager.save_lora(self.model, lora_path)
        
        # Save training state
        state_path = os.path.join(ckpt_dir, "training_state.pt")
        torch.save({
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
        }, state_path)
        
        print(f"[Checkpoint] Saved to {ckpt_dir}")
    
    def _load_checkpoint(self, ckpt_path: str):
        """Load training checkpoint."""
        print(f"[Checkpoint] Loading from {ckpt_path}")
        
        # Load LoRA weights
        lora_weights_path = os.path.join(ckpt_path, "lora_weights.pt")
        if os.path.exists(lora_weights_path):
            self.lora_manager.load_lora(self.model, lora_weights_path)
        
        # Load training state
        state_path = os.path.join(ckpt_path, "training_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location='cpu')
            self.global_step = state['global_step']
            self.epoch = state['epoch']
            self.optimizer.load_state_dict(state['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(state['lr_scheduler_state_dict'])
        
        print(f"[Checkpoint] Resumed from step {self.global_step}")
