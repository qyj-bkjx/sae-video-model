"""
Logging system - supports WandB + TensorBoard
"""

import os
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """Base logger class"""
    
    @abstractmethod
    def log(self, metrics: Dict[str, Any], step: int):
        pass
    
    @abstractmethod
    def log_images(self, images: Dict[str, Any], step: int):
        pass
    
    @abstractmethod
    def finish(self):
        pass


class WandBLogger(BaseLogger):
    """Weights & Biases logger"""
    
    def __init__(
        self,
        project: str,
        name: str = None,
        config: Dict = None,
        tags: list = None,
        notes: str = None,
        dir: str = "./wandb",
        resume: str = None,
        id: str = None,
    ):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            raise ImportError("Please install wandb: pip install wandb")
        
        os.makedirs(dir, exist_ok=True)
        
        self.run = wandb.init(
            project=project,
            name=name,
            config=config,
            tags=tags,
            notes=notes,
            dir=dir,
            resume=resume,
            id=id,
        )
        print(f"[WandB] Initialized: {self.run.url}")
    
    def log(self, metrics: Dict[str, Any], step: int):
        self.wandb.log(metrics, step=step)
    
    def log_images(self, images: Dict[str, Any], step: int):
        """Log images/videos"""
        log_dict = {}
        for name, img in images.items():
            if isinstance(img, list):
                log_dict[name] = self.wandb.Video(img, fps=24, format="mp4")
            else:
                log_dict[name] = self.wandb.Image(img)
        self.wandb.log(log_dict, step=step)
    
    def log_audio(self, audios: Dict[str, Any], step: int, sample_rate: int = 48000):
        """Log audio"""
        log_dict = {}
        for name, audio in audios.items():
            log_dict[name] = self.wandb.Audio(audio.cpu().numpy(), sample_rate=sample_rate)
        self.wandb.log(log_dict, step=step)
    
    def finish(self):
        self.wandb.finish()


class TensorBoardLogger(BaseLogger):
    """TensorBoard logger"""
    
    def __init__(self, log_dir: str = "./tensorboard"):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("Please install tensorboard: pip install tensorboard")
        
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"[TensorBoard] Log dir: {log_dir}")
    
    def log(self, metrics: Dict[str, Any], step: int):
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)
    
    def log_images(self, images: Dict[str, Any], step: int):
        """Log images"""
        import torch
        import numpy as np
        
        for name, img in images.items():
            if isinstance(img, list):
                img = img[0]
            
            if hasattr(img, 'numpy'):
                img = img.numpy()
            elif hasattr(img, 'convert'):
                img = np.array(img)
            
            if img.ndim == 3 and img.shape[-1] in [1, 3, 4]:
                img = img.transpose(2, 0, 1)  # HWC -> CHW
            
            self.writer.add_image(name, img, step)
    
    def log_audio(self, audios: Dict[str, Any], step: int, sample_rate: int = 48000):
        """Log audio"""
        for name, audio in audios.items():
            self.writer.add_audio(name, audio.cpu(), step, sample_rate=sample_rate)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram"""
        self.writer.add_histogram(tag, values, step)
    
    def finish(self):
        self.writer.close()


class CompositeLogger(BaseLogger):
    """Composite multiple loggers"""
    
    def __init__(self, loggers: list):
        self.loggers = loggers
    
    def log(self, metrics: Dict[str, Any], step: int):
        for logger in self.loggers:
            logger.log(metrics, step)
    
    def log_images(self, images: Dict[str, Any], step: int):
        for logger in self.loggers:
            logger.log_images(images, step)
    
    def log_audio(self, audios: Dict[str, Any], step: int, sample_rate: int = 48000):
        for logger in self.loggers:
            if hasattr(logger, 'log_audio'):
                logger.log_audio(audios, step, sample_rate)
    
    def finish(self):
        for logger in self.loggers:
            logger.finish()


class DummyLogger(BaseLogger):
    """Dummy logger (for non-main processes)"""
    
    def log(self, metrics: Dict[str, Any], step: int):
        pass
    
    def log_images(self, images: Dict[str, Any], step: int):
        pass
    
    def finish(self):
        pass


def build_logger(
    logger_type: str = "wandb",
    is_main_process: bool = True,
    **kwargs
) -> BaseLogger:
    """
    Build logger
    
    Args:
        logger_type: "wandb", "tensorboard", "both", "none"
        is_main_process: Whether is main process (returns DummyLogger for non-main processes)
        **kwargs: Passed to specific logger
    """
    if not is_main_process:
        return DummyLogger()
    
    if logger_type == "none":
        return DummyLogger()
    elif logger_type == "wandb":
        return WandBLogger(**kwargs)
    elif logger_type == "tensorboard":
        return TensorBoardLogger(**kwargs)
    elif logger_type == "both":
        loggers = [
            WandBLogger(**{k: v for k, v in kwargs.items() if k not in ['log_dir']}),
            TensorBoardLogger(log_dir=kwargs.get('log_dir', './tensorboard'))
        ]
        return CompositeLogger(loggers)
    else:
        raise ValueError(f"Unknown logger type: {logger_type}")

