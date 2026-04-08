"""
LoRA (Low-Rank Adaptation) utilities

Supports:
- Dynamically inject LoRA layers into existing models
- Save/load LoRA weights
- Compatible with PEFT format
"""

import os
import re
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union

from diffusers import DiffusionPipeline


class LoRALinear(nn.Module):
    """
    LoRA linear layer
    
    Implementation: h = Wx + (BA)x * (alpha / rank)
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        original_dtype = next(original_layer.parameters()).dtype
        
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        self.lora_A = self.lora_A.to(dtype=original_dtype)
        self.lora_B = self.lora_B.to(dtype=original_dtype)
        
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
            nn.init.zeros_(self.lora_B.weight)
        
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def to(self, *args, **kwargs):
        """Rewrite to method to ensure LoRA layer is also correctly converted"""
        result = super().to(*args, **kwargs)
        
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']
            self.lora_A.to(dtype=dtype)
            self.lora_B.to(dtype=dtype)
        elif len(args) > 0 and isinstance(args[0], torch.dtype):
            dtype = args[0]
            self.lora_A.to(dtype=dtype)
            self.lora_B.to(dtype=dtype)
        
        return result
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original_layer(x)
        
        x_dropped = self.lora_dropout(x)
        input_dtype = x_dropped.dtype
        lora_dtype = self.lora_A.weight.dtype
        
        if input_dtype != lora_dtype:
            x_dropped_converted = x_dropped.to(dtype=lora_dtype)
            lora_output = self.lora_B(self.lora_A(x_dropped_converted))
            # Cast back to input dtype.
            lora_output = lora_output.to(dtype=input_dtype)
        else:
            lora_output = self.lora_B(self.lora_A(x_dropped))
        
        if lora_output.dtype != result.dtype:
            lora_output = lora_output.to(dtype=result.dtype)
        
        return result + lora_output * self.scaling
    
    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into original layer"""
        merged = nn.Linear(
            self.original_layer.in_features,
            self.original_layer.out_features,
            bias=self.original_layer.bias is not None
        )
        
        delta_weight = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        merged.weight.data = self.original_layer.weight.data + delta_weight
        
        if self.original_layer.bias is not None:
            merged.bias.data = self.original_layer.bias.data
        
        return merged


def inject_lora_to_model(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: List[str] = None,
    exclude_modules: List[str] = None,
) -> Dict[str, LoRALinear]:
    """
    Inject LoRA layers into model
    
    Args:
        model: Target model
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: Dropout rate
        target_modules: Target module name patterns list, e.g. ["q", "k", "v", "o", "proj"]
                        Supports regular matching
        exclude_modules: Excluded module name patterns list, e.g. ["time_projection", "time_embedding"]
                         Supports regular matching
    
    Returns:
        Dictionary of injected LoRA layers
    """
    if target_modules is None:
        target_modules = ["q", "k", "v", "o", "to_q", "to_k", "to_v", "to_out"]
    if exclude_modules is None:
        exclude_modules = ["time_projection", "time_embedding"]
    
    patterns = [re.compile(f".*{pattern}.*") for pattern in target_modules]
    exclude_patterns = [re.compile(f".*{pattern}.*") for pattern in exclude_modules]
    
    lora_layers = {}
    
    def is_excluded(name: str) -> bool:
        """Check if module is in exclude list"""
        for exclude_pattern in exclude_patterns:
            if exclude_pattern.match(name):
                return True
        return False
    
    def should_inject(name: str) -> bool:
        """Check if should inject LoRA"""
        if is_excluded(name):
            return False
        for pattern in patterns:
            if pattern.match(name):
                return True
        return False
    
    def inject_recursive(module: nn.Module, prefix: str = ""):
        """Recursive injection of LoRA"""
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if is_excluded(full_name):
                continue
            
            if isinstance(child, nn.Linear) and should_inject(full_name):
                lora_layer = LoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(module, name, lora_layer)
                lora_layers[full_name] = lora_layer
            else:
                inject_recursive(child, full_name)
    
    inject_recursive(model)
    
    print(f"[LoRA] Injected {len(lora_layers)} LoRA layers")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[LoRA] Total parameters: {total_params / 1e6:.2f}M")
    print(f"[LoRA] Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"[LoRA] Trainable ratio: {trainable_params / total_params * 100:.2f}%")

    freeze_non_lora_params(model)
    return lora_layers


def freeze_non_lora_params(model: nn.Module):
    """Freeze non-LoRA parameters"""
    for name, param in model.named_parameters():
        if not ("lora_" in name):
            param.requires_grad = False
            
    # Count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] After freezing: {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total")
    return model

def save_lora_weights(
    model: nn.Module,
    save_path: str,
    module_names: List[str] = None,
    format: str = "peft",  # "peft" or "diffsynth"
):
    """
    Save LoRA weights (non-FSDP mode)
    
    Args:
        model: Model containing LoRA layers
        save_path: Save path
        module_names: List of module names to save
        format: Save format ("peft" or "diffsynth")
    """
    os.makedirs(save_path, exist_ok=True)
    
    lora_state_dict = {}
    target_modules = ["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"]

    loaded_count = 0
    for module_name in target_modules:
        module = getattr(model, module_name)
        for name, module in module.named_modules():
            if module_names:
                is_target = any(m in name for m in module_names)
                if not is_target:
                    continue
            
            if isinstance(module, LoRALinear):
                if format == "peft":
                    lora_state_dict[f"{module_name}.{name}.lora_A.weight"] = module.lora_A.weight.data.cpu()
                    lora_state_dict[f"{module_name}.{name}.lora_B.weight"] = module.lora_B.weight.data.cpu()
                else:
                    lora_state_dict[f"{module_name}.{name}.lora_A.default.weight"] = module.lora_A.weight.data.cpu()
                    lora_state_dict[f"{module_name}.{name}.lora_B.default.weight"] = module.lora_B.weight.data.cpu()
    
    config = {
        "rank": next(iter(model.modules())).__class__.__name__ if lora_state_dict else 16,
        "alpha": 16.0,
        "target_modules": module_names or [],
    }
    
    torch.save(lora_state_dict, os.path.join(save_path, "lora_weights.pt"))
    torch.save(config, os.path.join(save_path, "lora_config.pt"))
    
    print(f"[LoRA] Saved {len(lora_state_dict)} tensors to {save_path}")


def save_lora_weights_fsdp(
    full_state_dict: Dict[str, torch.Tensor],
    save_path: str,
    module_names: List[str] = None,
    format: str = "peft",
    is_main_process: bool = True,
):
    """
    Save LoRA weights (FSDP mode)
    
    Args:
        full_state_dict: Full state dictionary returned by accelerator.get_state_dict()
        save_path: Save path
        module_names: List of module names to save
        format: Save format ("peft" or "diffsynth")
        is_main_process: Whether is main process (only main process needs to save)
    """
    if not is_main_process:
        return
    
    os.makedirs(save_path, exist_ok=True)
    
    lora_state_dict = {}
    
    for key, value in full_state_dict.items():
        if "lora_A.weight" not in key and "lora_B.weight" not in key:
            continue
        
        if module_names:
            is_target = any(m in key for m in module_names)
            if not is_target:
                continue
        
        if format == "peft":
            lora_state_dict[key] = value.cpu()
        else:
            new_key = key.replace(".lora_A.weight", ".lora_A.default.weight")
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
            lora_state_dict[new_key] = value.cpu()
    
    config = {
        "rank": 16,
        "alpha": 16.0,
        "target_modules": module_names or [],
    }
    
    torch.save(lora_state_dict, os.path.join(save_path, "lora_weights.pt"))
    torch.save(config, os.path.join(save_path, "lora_config.pt"))
    
    print(f"[LoRA-FSDP] Saved {len(lora_state_dict)} tensors to {save_path}")


def load_lora_weights(
    model: DiffusionPipeline,
    load_path: str,
    alpha: float = None,
):
    """
    Load LoRA weights
    
    Args:
        model: Target model (already injected LoRA layers)
        load_path: Weight path
        alpha: Optional alpha scaling factor (override saved value)
    """
    weights_path = os.path.join(load_path, "lora_weights.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LoRA weights not found: {weights_path}")
    
    lora_state_dict = torch.load(weights_path, map_location="cpu")
    target_modules = ["video_dit", "video_dit_2", "audio_dit", "dual_tower_bridge"]

    loaded_count = 0
    for module_name in target_modules:
        module = getattr(model, module_name)
        for name, module in module.named_modules():
            if isinstance(module, LoRALinear):
                a_key = f"{module_name}.{name}.lora_A.weight"
                b_key = f"{module_name}.{name}.lora_B.weight"
                
                if a_key not in lora_state_dict:
                    a_key = f"{module_name}.{name}.lora_A.default.weight"
                    b_key = f"{module_name}.{name}.lora_B.default.weight"
                
                if a_key in lora_state_dict and b_key in lora_state_dict:
                    module.lora_A.weight.data = lora_state_dict[a_key].to(module.lora_A.weight.device)
                    module.lora_B.weight.data = lora_state_dict[b_key].to(module.lora_B.weight.device)
                    loaded_count += 1
                    
                    if alpha is not None:
                        module.alpha = alpha
                        module.scaling = alpha / module.rank
    
    print(f"[LoRA] Loaded {loaded_count} LoRA layers from {load_path}")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into original model
    
    Args:
        model: Model containing LoRA layers
    
    Returns:
        Merged model
    """
    merged_count = 0
    
    def merge_recursive(module: nn.Module, parent: nn.Module = None, name: str = ""):
        nonlocal merged_count
        
        for child_name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                merged_layer = child.merge_weights()
                setattr(module, child_name, merged_layer)
                merged_count += 1
            else:
                merge_recursive(child, module, child_name)
    
    merge_recursive(model)
    print(f"[LoRA] Merged {merged_count} LoRA layers")
    
    return model


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters"""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend(module.lora_A.parameters())
            params.extend(module.lora_B.parameters())
    return params


def count_lora_params(model: nn.Module) -> Dict[str, int]:
    """Count LoRA parameters"""
    total = 0
    lora_count = 0
    
    for name, param in model.named_parameters():
        total += param.numel()
        if "lora_" in name:
            lora_count += param.numel()
    
    return {
        "total_params": total,
        "lora_params": lora_count,
        "lora_ratio": lora_count / total if total > 0 else 0,
    }


# ============================================================
# PEFT compatible layers
# ============================================================

def from_peft_model(peft_model) -> Dict[str, torch.Tensor]:
    """
    Extract LoRA weights from PEFT model
    
    Args:
        peft_model: PEFT wrapped model
    
    Returns:
        LoRA state dictionary
    """
    try:
        from peft import get_peft_model_state_dict
        return get_peft_model_state_dict(peft_model)
    except ImportError:
        raise ImportError("Please install peft: pip install peft")


def to_peft_config(
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: List[str] = None,
):
    """
    Create PEFT LoRA configuration
    
    Returns:
        PEFT LoraConfig object
    """
    try:
        from peft import LoraConfig
    except ImportError:
        raise ImportError("Please install peft: pip install peft")
    
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=None,
    )

