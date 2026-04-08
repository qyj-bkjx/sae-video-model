"""
MOVA LoRA inference Pipeline
"""

import os
import torch
from typing import Optional, List, Union, Any

from mova.diffusion.pipelines.pipeline_mova import MOVA

ACCELERATE_TRAINER_TRAINED = os.getenv('ACCELERATE_TRAINER_TRAINED', '1')
if ACCELERATE_TRAINER_TRAINED:
    from mova.engine.trainer.accelerate.lora_utils import inject_lora_to_model, load_lora_weights, LoRALinear
else:
    from mova.engine.trainer.low_resource.lora_layers import inject_lora_into_model, load_lora_state_dict, LoRALinear
from mova.registry import DIFFUSION_PIPELINES

from diffusers.models.autoencoders import AutoencoderKLWan
from mova.diffusion.models.dac_vae import DAC
from transformers.models import UMT5EncoderModel, T5TokenizerFast

from mova.diffusion.models import (
    WanAudioModel, WanModel, sinusoidal_embedding_1d, 
) 
from mova.diffusion.models.interactionv2 import DualTowerConditionalBridge

class MOVALoRA(MOVA):
    """
    LoRA-enabled MOVA inference pipeline
    """
    
    def __init__(self,
                video_vae: AutoencoderKLWan,
                audio_vae: DAC,
                text_encoder: UMT5EncoderModel,
                tokenizer: T5TokenizerFast,
                scheduler: Any,
                video_dit: WanModel,
                video_dit_2: WanModel,
                audio_dit: WanAudioModel,
                dual_tower_bridge: DualTowerConditionalBridge,
                audio_vae_type: str = "dac", 
                boundary_ratio: float = 0.9,):
        super().__init__(
            video_vae=video_vae,
            audio_vae=audio_vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            video_dit=video_dit,
            video_dit_2=video_dit_2,
            audio_dit=audio_dit,
            dual_tower_bridge=dual_tower_bridge,
            audio_vae_type=audio_vae_type,
            boundary_ratio=boundary_ratio,
            )
        self._lora_injected = False
        self._lora_layers = {}
    
    @classmethod
    def from_pretrained_with_lora(
        cls,
        pretrained_path: str,
        lora_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        lora_alpha: float = None,
        lora_modules: List[str] = None,
    ) -> "MOVALoRA":
        """
        Load from pretrained model and LoRA weights
        
        Args:
            pretrained_path: Path to pretrained model
            lora_path: Path to LoRA weights (contains lora_weights.pt and lora_config.pt)
            device: Device
            torch_dtype: Data type
            lora_alpha: LoRA alpha scaling factor (optional, overrides saved value)
            lora_modules: List of modules to inject LoRA (optional, overrides saved value)
        
        Returns:
            MOVALoRA: Pipeline with LoRA loaded
        """
        print(f"[MOVALoRA] Loading base model from {pretrained_path}")

        base_pipeline = cls.from_pretrained(
            pretrained_path,
            torch_dtype=torch_dtype
        )

        base_pipeline.__class__ = cls
        if not hasattr(base_pipeline, "_lora_injected"):
            base_pipeline._lora_injected = False
        if not hasattr(base_pipeline, "_lora_layers"):
            base_pipeline._lora_layers = {}

        pipeline: "MOVALoRA" = base_pipeline  
        pipeline.load_lora(
            lora_path=lora_path,
            alpha=lora_alpha,
            target_modules=lora_modules,
        )
        
        if device != "cpu":
            pipeline = pipeline.to(device)
        
        return pipeline
    
    def load_lora(
        self,
        lora_path: str,
        alpha: float = None,
        target_modules: List[str] = None,
    ):
        """
        Load LoRA weights.
        
        Args:
            lora_path: LoRA weights path
            alpha: LoRA alpha scaling factor (optional)
            target_modules: List of modules to inject LoRA (optional)
        """
        print(f"[MOVALoRA] Loading LoRA weights from {lora_path}")
        
        config_path = os.path.join(lora_path, "lora_config.pt")
        if os.path.exists(config_path):
            config = torch.load(config_path, map_location="cpu")
            print(f"[MOVALoRA] LoRA config: {config}")
        else:
            config = {"rank": 16, "alpha": 16.0, "target_modules": []}
            print(f"[MOVALoRA] No config found, using defaults: {config}")
        
        rank = config.get("rank", 16)
        lora_alpha = alpha if alpha is not None else config.get("alpha", 16.0)
        modules = target_modules if target_modules is not None else config.get("target_modules", [])
        
        if not modules:
            modules = ["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"]
        
        if not self._lora_injected:
            self._inject_lora_layers(rank=rank, alpha=lora_alpha, target_modules=modules)
        
        load_lora_weights(self, lora_path, alpha=lora_alpha)
        
        print(f"[MOVALoRA] LoRA loaded successfully")
    
    def _inject_lora_layers(
        self,
        rank: int = 16,
        alpha: float = 16.0,
        target_modules: List[str] = None,
    ):
        """
        Inject LoRA layers into model
        
        Args:
            rank: LoRA rank
            alpha: LoRA alpha
            target_modules: List of module names
        """
        if self._lora_injected:
            print("[MOVALoRA] LoRA already injected, skipping")
            return
        
        if target_modules is None:
            target_modules = ["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"]
        

        target_layer_names = ["q", "k", "v", "o", "to_q", "to_k", "to_v", "proj"]
        
        total_injected = 0
        
        for module_name in target_modules:
            if hasattr(self, module_name):
                module = getattr(self, module_name)
                if module is not None:
                    lora_layers = inject_lora_to_model(
                        module,
                        rank=rank,
                        alpha=alpha,
                        target_modules=target_layer_names,
                    )
                    self._lora_layers.update(lora_layers)
                    total_injected += len(lora_layers)
                    print(f"[MOVALoRA] Injected {len(lora_layers)} LoRA layers to {module_name}")
        
        self._lora_injected = True
        print(f"[MOVALoRA] Total injected: {total_injected} LoRA layers")
    
    def merge_lora_weights(self):
        """
        Merge LoRA weights into base model
        
        After merging, LoRA layers will be removed, and the model will revert to a normal model, but with LoRA effects.
        This is useful for inference optimization because after merging, no additional LoRA computation is needed.
        """
        if not self._lora_injected:
            print("[MOVALoRA] No LoRA layers to merge")
            return
        
        merged_count = 0
        
        def merge_recursive(parent_module, parent_name=""):
            nonlocal merged_count
            for name, child in list(parent_module.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name
                
                if isinstance(child, LoRALinear):
                    merged_layer = child.merge_weights()
                    setattr(parent_module, name, merged_layer)
                    merged_count += 1
                else:
                    merge_recursive(child, full_name)
        
        merge_recursive(self)
        
        self._lora_injected = False
        self._lora_layers = {}
        
        print(f"[MOVALoRA] Merged {merged_count} LoRA layers into base model")
    
    def unload_lora(self):
        """
        Unload LoRA layers, restore original model
        
        Note: This will lose LoRA weights, if you need to preserve effects, please use merge_lora_weights()
        """
        if not self._lora_injected:
            print("[MOVALoRA] No LoRA layers to unload")
            return
        
        unloaded_count = 0
        
        def unload_recursive(parent_module, parent_name=""):
            nonlocal unloaded_count
            for name, child in list(parent_module.named_children()):
                full_name = f"{parent_name}.{name}" if parent_name else name
                
                if isinstance(child, LoRALinear):
                    setattr(parent_module, name, child.original_layer)
                    unloaded_count += 1
                else:
                    unload_recursive(child, full_name)
        
        unload_recursive(self)
        
        self._lora_injected = False
        self._lora_layers = {}
        
        print(f"[MOVALoRA] Unloaded {unloaded_count} LoRA layers")
    
    def set_lora_scale(self, scale: float):
        """
        Set LoRA scale factor
        
        Args:
            scale: New scale factor (0.0 = disable LoRA, 1.0 = normal strength)
        """
        if not self._lora_injected:
            print("[MOVALoRA] No LoRA layers to scale")
            return
        
        for name, module in self.named_modules():
            if isinstance(module, LoRALinear):
                original_scaling = module.alpha / module.rank
                module.scaling = original_scaling * scale
        
        print(f"[MOVALoRA] Set LoRA scale to {scale}")


@DIFFUSION_PIPELINES.register_module(name="MOVALoRA_from_pretrained")
def MOVALoRA_from_pretrained(
    from_pretrained: str,
    lora_path: str = None,
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.bfloat16,
    lora_alpha: float = None,
) -> MOVALoRA:
    """
    Registered factory function for loading from config file
    
    Args:
        from_pretrained: Path to base model
        lora_path: Path to LoRA weights (optional)
        device: Device
        torch_dtype: Data type
        lora_alpha: LoRA alpha scaling factor
    
    Returns:
        MOVALoRA: pipeline instance
    """
    if lora_path:
        return MOVALoRA.from_pretrained_with_lora(
            pretrained_path=from_pretrained,
            lora_path=lora_path,
            device=device,
            torch_dtype=torch_dtype,
            lora_alpha=lora_alpha,
        )
    else:
        pipeline = MOVALoRA.from_pretrained(
            from_pretrained,
            torch_dtype=torch_dtype,
        )
        if device != "cpu":
            pipeline = pipeline.to(device)
        return pipeline
