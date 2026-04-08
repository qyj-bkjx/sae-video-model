"""
LoRA (Low-Rank Adaptation) Layers

This module implements LoRA layers for efficient fine-tuning of large models.
LoRA decomposes weight updates into low-rank matrices, significantly reducing
the number of trainable parameters.

Reference: https://arxiv.org/abs/2106.09685
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Set, Tuple, Union
import math
import re
import os

DEBUG_OUTPUT = os.getenv('DEBUG_OUTPUT', '0')   
DEBUG_OUTPUT = int(DEBUG_OUTPUT)

def debug_print(*args, **kwargs):
    """Print only if DEBUG_OUTPUT is enabled."""
    if DEBUG_OUTPUT:
        kwargs.setdefault('flush', True)
        print(*args, **kwargs)

class LoRALinear(nn.Module):
    """
    LoRA layer that wraps a frozen linear layer with trainable low-rank adapters.
    
    The output is: y = Wx + (BA)x * (alpha / r)
    where W is frozen, B and A are trainable low-rank matrices.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (r)
            alpha: LoRA scaling factor
            dropout: Dropout rate for LoRA
            bias: Whether to use bias
            dtype: Data type for weights
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Frozen original weight (will be set from pretrained)
        self.register_buffer('weight', torch.zeros(out_features, in_features, dtype=dtype))
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=dtype))
        else:
            self.register_buffer('bias', None)
        
        # Trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype))
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        """Initialize LoRA matrices - A with Kaiming, B with zeros."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        # Original linear transformation
        result = F.linear(x, self.weight, self.bias)
        
        # LoRA delta: BA * x * scaling
        lora_input = self.dropout(x)
        lora_output = F.linear(F.linear(lora_input, self.lora_A), self.lora_B)
        result = result + lora_output * self.scaling
        
        return result
    
    def merge_lora(self) -> torch.Tensor:
        """
        Merge LoRA weights into the main weight for inference.
        
        Returns:
            Merged weight tensor
        """
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        return self.weight + delta_w
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 8, alpha: float = 8.0,
                    dropout: float = 0.0, dtype: torch.dtype = None) -> 'LoRALinear':
        """
        Create a LoRALinear from an existing Linear layer.
        
        Args:
            linear: Source nn.Linear layer
            rank: LoRA rank
            alpha: LoRA alpha
            dropout: LoRA dropout
            dtype: Override dtype (if None, use source weight dtype)
            
        Returns:
            LoRALinear with copied frozen weights
        """
        if dtype is None:
            dtype = linear.weight.dtype
        
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
            dtype=dtype,
        )
        
        # Copy frozen weights
        lora_linear.weight.copy_(linear.weight.detach().to(dtype))
        if linear.bias is not None:
            lora_linear.bias.copy_(linear.bias.detach().to(dtype))
        
        return lora_linear


class LoRAConv3d(nn.Module):
    """
    LoRA layer for Conv3d, using 1x1x1 convolutions for low-rank adaptation.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Frozen original weight
        self.register_buffer('weight', torch.zeros(
            out_channels, in_channels, *kernel_size, dtype=dtype))
        if bias:
            self.register_buffer('bias', torch.zeros(out_channels, dtype=dtype))
        else:
            self.register_buffer('bias', None)
        
        # LoRA uses 1x1x1 convolutions
        self.lora_A = nn.Parameter(torch.zeros(rank, in_channels, 1, 1, 1, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, rank, 1, 1, 1, dtype=dtype))
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original convolution
        result = F.conv3d(x, self.weight, self.bias, self.stride, self.padding)
        
        # LoRA convolution (1x1x1)
        lora_input = self.dropout(x)
        lora_out = F.conv3d(lora_input, self.lora_A, stride=1, padding=0)
        lora_out = F.conv3d(lora_out, self.lora_B, stride=1, padding=0)
        
        # Resize if needed (due to stride/kernel mismatch)
        if lora_out.shape != result.shape:
            lora_out = F.interpolate(lora_out, size=result.shape[2:], mode='trilinear', align_corners=False)
        
        return result + lora_out * self.scaling


def inject_lora_into_model(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: List[str] = None,
    exclude_modules: List[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[nn.Module, Dict[str, nn.Module]]:
    """
    Inject LoRA layers into a model.
    
    This function replaces specified Linear/Conv layers with LoRA variants,
    freezes the original weights, and makes only LoRA parameters trainable.
    
    Args:
        model: The model to modify
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout
        target_modules: List of module name patterns to apply LoRA
                       (e.g., ["q", "k", "v", "o", "proj"])
        exclude_modules: List of module name patterns to exclude
        dtype: Data type for LoRA weights
        
    Returns:
        Tuple of (modified_model, dict of replaced modules)
    """
    if target_modules is None:
        target_modules = ["q", "k", "v", "o", "proj", "to_q", "to_k", "to_v", "to_out"]
    
    if exclude_modules is None:
        exclude_modules = []
    
    replaced_modules = {}
    
    def _should_replace(name: str) -> bool:
        """Check if module should be replaced with LoRA."""
        # Check exclusions first
        for exclude in exclude_modules:
            if exclude in name:
                return False
        
        # Check if matches target patterns
        for target in target_modules:
            # Match module name (last part of the path)
            module_name = name.split('.')[-1]
            if module_name == target or name.endswith('.' + target):
                return True
            # Also match with regex for more flexibility
            if re.search(rf'(^|\.){target}$', name):
                return True
        
        return False
    
    def _replace_module(parent: nn.Module, name: str, module: nn.Module) -> Optional[nn.Module]:
        """Replace a module with its LoRA variant."""
        if isinstance(module, nn.Linear):
            lora_module = LoRALinear.from_linear(
                module, rank=rank, alpha=alpha, dropout=dropout, dtype=dtype
            )
            setattr(parent, name, lora_module)
            return lora_module
        # Add more module types as needed (Conv2d, Conv3d, etc.)
        return None
    
    # Iterate through all modules and replace matching ones
    modules_to_replace = []
    
    for full_name, module in model.named_modules():
        if _should_replace(full_name):
            if isinstance(module, (nn.Linear,)):
                modules_to_replace.append((full_name, module))
    
    print(f"[LoRA] Found {len(modules_to_replace)} modules to inject LoRA")
    
    for full_name, module in modules_to_replace:
        # Get parent module and attribute name
        parts = full_name.rsplit('.', 1)
        if len(parts) == 2:
            parent_name, attr_name = parts
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            attr_name = full_name
        
        new_module = _replace_module(parent, attr_name, module)
        if new_module is not None:
            replaced_modules[full_name] = new_module
            debug_print(f"[LoRA] Replaced: {full_name} (in={module.in_features}, out={module.out_features}, rank={rank})")
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"[LoRA] Total parameters: {total_params / 1e6:.2f}M")
    print(f"[LoRA] Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"[LoRA] Trainable ratio: {trainable_params / total_params * 100:.2f}%")
    
    return model, replaced_modules


def freeze_non_lora_params(model: nn.Module):
    """
    Freeze all parameters except LoRA parameters.
    
    LoRA parameters are identified by containing 'lora_' in their name.
    
    Args:
        model: The model to freeze
    """
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # Count
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] After freezing: {trainable / 1e6:.2f}M trainable / {total / 1e6:.2f}M total")


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters from model state dict.
    
    Args:
        model: The model with LoRA layers
        
    Returns:
        State dict containing only LoRA parameters
    """
    lora_state_dict = {}
    for name, param in model.state_dict().items():
        if 'lora_' in name:
            lora_state_dict[name] = param
    return lora_state_dict


def load_lora_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor], 
                          strict: bool = False):
    """
    Load LoRA parameters into a model.
    
    Args:
        model: The model with LoRA layers
        state_dict: State dict with LoRA parameters
        strict: Whether to strictly enforce that keys match
    """
    model_state = model.state_dict()
    
    loaded_keys = []
    for name, param in state_dict.items():
        if name in model_state:
            if model_state[name].shape == param.shape:
                model_state[name].copy_(param)
                loaded_keys.append(name)
            else:
                print(f"[LoRA] Shape mismatch for {name}: {model_state[name].shape} vs {param.shape}")
        elif strict:
            raise KeyError(f"Key {name} not found in model state dict")
    
    print(f"[LoRA] Loaded {len(loaded_keys)} LoRA parameters")


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge LoRA weights into base weights for efficient inference.
    
    After merging, LoRA layers behave like regular layers without
    the additional LoRA computation.
    
    Args:
        model: The model with LoRA layers
        
    Returns:
        Model with merged weights
    """
    merged_count = 0
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # Merge weights
            merged_weight = module.merge_lora()
            module.weight.copy_(merged_weight)
            # Zero out LoRA
            module.lora_A.zero_()
            module.lora_B.zero_()
            merged_count += 1
    
    print(f"[LoRA] Merged {merged_count} LoRA layers")
    return model


class LoRAManager:
    """
    Manager class for LoRA training utilities.
    
    Provides convenience methods for:
    - Injecting LoRA into models
    - Saving/loading LoRA weights
    - Managing LoRA training
    """
    
    def __init__(
        self,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
        target_modules: List[str] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.dtype = dtype
        
        self.lora_modules: Dict[str, nn.Module] = {}
    
    def inject_lora(self, model: nn.Module, train_modules: List[str] = None) -> nn.Module:
        """
        Inject LoRA into specified modules of the model.
        
        Args:
            model: The model to modify
            train_modules: List of top-level module names to inject LoRA into
                          (e.g., ["video_dit", "audio_dit"])
        
        Returns:
            Modified model
        """
        if train_modules is None:
            # Inject into the whole model
            model, self.lora_modules = inject_lora_into_model(
                model,
                rank=self.rank,
                alpha=self.alpha,
                dropout=self.dropout,
                target_modules=self.target_modules,
                dtype=self.dtype,
            )
        else:
            # Only inject into specified modules
            all_replaced = {}
            for module_name in train_modules:
                if hasattr(model, module_name):
                    submodule = getattr(model, module_name)
                    submodule, replaced = inject_lora_into_model(
                        submodule,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                        target_modules=self.target_modules,
                        dtype=self.dtype,
                    )
                    # Prefix with module name
                    all_replaced.update({
                        f"{module_name}.{k}": v for k, v in replaced.items()
                    })
            self.lora_modules = all_replaced
        
        # Freeze non-LoRA parameters
        freeze_non_lora_params(model)
        
        return model
    
    def get_trainable_params(self, model: nn.Module) -> List[nn.Parameter]:
        """Get list of trainable (LoRA) parameters."""
        return [p for p in model.parameters() if p.requires_grad]
    
    def save_lora(self, model: nn.Module, path: str):
        """Save LoRA weights to file (non-FSDP version)."""
        state_dict = get_lora_state_dict(model)
        torch.save({
            'lora_state_dict': state_dict,
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'target_modules': self.target_modules,
        }, path)
        print(f"[LoRA] Saved to {path}")
    
    def save_lora_fsdp(
        self,
        full_state_dict: Dict[str, torch.Tensor],
        save_path: str,
        module_names: List[str] = None,
        is_main_process: bool = True,
    ):
        """
        Save LoRA weights from FSDP model (FSDP version).
        
        In FSDP mode, parameters are sharded across processes. This method
        extracts LoRA weights from a full state dict collected via 
        accelerator.get_state_dict().
        
        Args:
            full_state_dict: Full state dict from accelerator.get_state_dict()
            save_path: Directory path to save weights
            module_names: List of module names to filter (e.g., ["video_dit", "audio_dit"])
            is_main_process: Whether this is the main process (only main process saves)
        """
        # Only save on main process
        if not is_main_process:
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        lora_state_dict = {}
        
        # Extract LoRA weights from full state dict
        for key, value in full_state_dict.items():
            # Check if it's a LoRA weight
            if "lora_A" not in key and "lora_B" not in key:
                continue
            
            # Check if it's a target module
            if module_names:
                is_target = any(m in key for m in module_names)
                if not is_target:
                    continue
            
            lora_state_dict[key] = value.cpu()
        
        # Save weights and config
        lora_path = os.path.join(save_path, "lora_weights.pt")
        torch.save({
            'lora_state_dict': lora_state_dict,
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'target_modules': module_names or self.target_modules,
        }, lora_path)
        
        print(f"[LoRA-FSDP] Saved {len(lora_state_dict)} LoRA tensors to {save_path}")
    
    def load_lora(self, model: nn.Module, path: str) -> nn.Module:
        """Load LoRA weights from file."""
        checkpoint = torch.load(path, map_location='cpu')
        load_lora_state_dict(model, checkpoint['lora_state_dict'])
        return model
