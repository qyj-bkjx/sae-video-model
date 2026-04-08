"""
FP8 CPU Offload Manager

This module implements a memory-efficient weight management system that:
1. Stores model weights in FP8 format on CPU
2. Transfers weights to GPU and dequantizes only when needed for computation
3. Quantizes and offloads weights back to CPU after computation

This allows training very large models on limited GPU memory by leveraging CPU RAM.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Union, Any
from collections import OrderedDict
import weakref


# FP8 dtype availability check
FP8_AVAILABLE = hasattr(torch, 'float8_e4m3fn')
if not FP8_AVAILABLE:
    raise ValueError("FP8 not available, please check your PyTorch version")
FP8_DTYPE = torch.float8_e4m3fn


def quantize_to_fp8(tensor: torch.Tensor, to_cpu: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a tensor to FP8 format with scale factor.
    
    Memory-efficient version that moves to CPU first to avoid GPU memory spikes.
    
    Args:
        tensor: Input tensor in any dtype (typically bf16/fp32)
        to_cpu: If True, move tensor to CPU before quantizing (saves GPU memory)
        
    Returns:
        Tuple of (fp8_tensor, scale) where:
        - fp8_tensor: Quantized tensor in FP8 format (on CPU if to_cpu=True)
        - scale: Scale factor for dequantization
    """
    if tensor.numel() == 0:
        return tensor.to(FP8_DTYPE), torch.tensor(1.0, dtype=torch.float32)
    
    # Move to CPU first to avoid GPU memory spike during quantization
    if to_cpu and tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    
    # Compute scale based on the maximum absolute value
    # FP8 E4M3 has a max value of 448
    fp8_max = 448.0  # Max value for E4M3
    
    # Use in-place operations where possible to save memory
    with torch.no_grad():
        max_val = tensor.abs().max().item()  # Scalar, no extra tensor
        
        if max_val == 0:
            scale = torch.tensor(1.0, dtype=torch.float32)
        else:
            scale = torch.tensor(max_val / fp8_max, dtype=torch.float32)
        
        # Scale down - avoid creating FP32 copy if input is already float
        if tensor.dtype in (torch.float32, torch.float64):
            # Already float, just divide in-place style
            scaled_tensor = (tensor / scale.item()).clamp_(-fp8_max, fp8_max)
        else:
            # BF16/FP16: need to convert, but do it efficiently
            scaled_tensor = tensor.float().div_(scale.item()).clamp_(-fp8_max, fp8_max)
        
        if FP8_AVAILABLE:
            fp8_tensor = scaled_tensor.to(FP8_DTYPE)
        else:
            # Fallback: use fp16 with reduced precision simulation
            fp8_tensor = scaled_tensor.to(torch.float16)
        
        # Free the scaled tensor immediately
        del scaled_tensor
    
    return fp8_tensor, scale


def dequantize_from_fp8(fp8_tensor: torch.Tensor, scale: torch.Tensor, 
                         target_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """
    Dequantize a FP8 tensor back to target dtype.
    
    Args:
        fp8_tensor: FP8 quantized tensor
        scale: Scale factor used during quantization
        target_dtype: Target dtype for output (typically bf16)
        
    Returns:
        Dequantized tensor in target dtype
    """
    return (fp8_tensor.float() * scale.float()).to(target_dtype)


class FP8WeightWrapper:
    """
    Wrapper class that holds FP8 weights on CPU and provides
    methods for GPU loading/offloading.
    """
    
    def __init__(self, weight: torch.Tensor, name: str):
        """
        Initialize with a weight tensor.
        
        Args:
            weight: Original weight tensor
            name: Name for this weight (for debugging)
        """
        self.name = name
        self.original_dtype = weight.dtype
        self.original_shape = weight.shape
        self.original_device = weight.device
        
        # Quantize to FP8 and store on CPU
        fp8_weight, scale = quantize_to_fp8(weight.detach().cpu())
        self.fp8_weight_cpu = fp8_weight.pin_memory()  # Pin for faster transfer
        self.scale_cpu = scale.cpu()
        
        # GPU cache (will be populated on demand)
        self._gpu_weight = None
        self._gpu_device = None
        
    def to_gpu(self, device: torch.device, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """
        Transfer weight to GPU and dequantize.
        
        Args:
            device: Target GPU device
            dtype: Target dtype (typically bf16)
            
        Returns:
            Dequantized weight tensor on GPU
        """
        if self._gpu_weight is not None and self._gpu_device == device:
            return self._gpu_weight
        
        # Transfer FP8 weight to GPU
        fp8_gpu = self.fp8_weight_cpu.to(device, non_blocking=True)
        scale_gpu = self.scale_cpu.to(device, non_blocking=True)
        
        # Dequantize to target dtype
        self._gpu_weight = dequantize_from_fp8(fp8_gpu, scale_gpu, dtype)
        self._gpu_device = device
        
        return self._gpu_weight
    
    def offload_to_cpu(self):
        """
        Offload GPU weight back to CPU (free GPU memory).
        """
        if self._gpu_weight is not None:
            del self._gpu_weight
            self._gpu_weight = None
            self._gpu_device = None
            torch.cuda.empty_cache()
    
    def update_from_gpu(self, gpu_weight: torch.Tensor):
        """
        Update the FP8 CPU weight from a GPU tensor (e.g., after gradient update).
        
        Args:
            gpu_weight: Updated weight tensor on GPU
        """
        # Re-quantize and store on CPU
        fp8_weight, scale = quantize_to_fp8(gpu_weight.detach().cpu())
        self.fp8_weight_cpu = fp8_weight.pin_memory()
        self.scale_cpu = scale.cpu()
        self._gpu_weight = None
        self._gpu_device = None


class FP8OffloadHook:
    """
    Forward/backward hook for automatic weight loading and offloading.
    """
    
    def __init__(self, module: nn.Module, weight_wrappers: Dict[str, FP8WeightWrapper],
                 device: torch.device, dtype: torch.dtype = torch.bfloat16):
        """
        Args:
            module: The nn.Module to hook
            weight_wrappers: Dict mapping param names to FP8WeightWrapper
            device: Target GPU device
            dtype: Computation dtype
        """
        self.module_ref = weakref.ref(module)
        self.weight_wrappers = weight_wrappers
        self.device = device
        self.dtype = dtype
        self._original_params = {}
        
    def pre_forward(self, module, inputs):
        """Load weights to GPU before forward."""
        for name, wrapper in self.weight_wrappers.items():
            gpu_weight = wrapper.to_gpu(self.device, self.dtype)
            # Replace the parameter temporarily
            param = getattr(module, name)
            self._original_params[name] = param
            # Create a new parameter with GPU weight
            setattr(module, name, nn.Parameter(gpu_weight, requires_grad=param.requires_grad))
    
    def post_forward(self, module, inputs, outputs):
        """Offload weights after forward (but keep for backward if training)."""
        # Don't offload during training - backward needs the weights
        if not module.training:
            for name, wrapper in self.weight_wrappers.items():
                wrapper.offload_to_cpu()
                # Restore original (dummy) parameter
                if name in self._original_params:
                    setattr(module, name, self._original_params[name])
        return outputs
    
    def post_backward(self, module, grad_input, grad_output):
        """Offload weights after backward pass."""
        for name, wrapper in self.weight_wrappers.items():
            wrapper.offload_to_cpu()


class FP8CPUOffloadManager:
    """
    Manager class for FP8 CPU offloading of model weights.
    
    This manager:
    1. Converts all specified weights to FP8 and stores on CPU
    2. Installs hooks to load weights on-demand during forward
    3. Offloads weights after forward/backward passes
    """
    
    def __init__(self, device: torch.device, dtype: torch.dtype = torch.bfloat16,
                 exclude_patterns: List[str] = None):
        """
        Args:
            device: GPU device for computation
            dtype: Computation dtype (typically bf16)
            exclude_patterns: List of parameter name patterns to exclude from offloading
        """
        self.device = device
        self.dtype = dtype
        self.exclude_patterns = exclude_patterns or []
        
        self.weight_wrappers: Dict[str, FP8WeightWrapper] = {}
        self.hooks: List[Any] = []
        self.managed_modules: Dict[str, nn.Module] = {}
        
    def _should_offload(self, name: str) -> bool:
        """Check if a parameter should be offloaded based on exclude patterns."""
        for pattern in self.exclude_patterns:
            if pattern in name:
                return False
        return True
    
    def _get_submodule_and_param_name(self, model: nn.Module, full_name: str):
        """Get the submodule and parameter name from a full parameter path."""
        parts = full_name.split('.')
        param_name = parts[-1]
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        return module, param_name
    
    def prepare_model_for_offload(self, model: nn.Module, 
                                   target_modules: List[str] = None) -> nn.Module:
        """
        Prepare a model for FP8 CPU offloading.
        
        This method:
        1. Identifies all weight parameters in specified modules
        2. Creates FP8 wrappers for each weight
        3. Installs forward/backward hooks for automatic loading/offloading
        
        Args:
            model: The model to prepare
            target_modules: List of module names to offload (if None, offload all)
            
        Returns:
            The modified model
        """
        print(f"[FP8 Offload] Preparing model for FP8 CPU offloading...")
        print(f"[FP8 Offload] Device: {self.device}, Dtype: {self.dtype}")
        
        # Collect parameters to offload
        params_to_offload = []
        
        for name, param in model.named_parameters():
            # Skip if not in target modules
            if target_modules:
                module_match = False
                for target in target_modules:
                    if name.startswith(target):
                        module_match = True
                        break
                if not module_match:
                    continue
            
            # Skip if in exclude patterns
            if not self._should_offload(name):
                continue
            
            # Only offload large weight parameters (skip biases, small params)
            if param.numel() > 1024 and 'weight' in name:
                params_to_offload.append((name, param))
        
        print(f"[FP8 Offload] Found {len(params_to_offload)} parameters to offload")
        
        # Create FP8 wrappers for each parameter
        total_original_size = 0
        total_fp8_size = 0
        
        for name, param in params_to_offload:
            original_size = param.numel() * param.element_size()
            total_original_size += original_size
            
            wrapper = FP8WeightWrapper(param, name)
            self.weight_wrappers[name] = wrapper
            
            fp8_size = wrapper.fp8_weight_cpu.numel() * wrapper.fp8_weight_cpu.element_size()
            total_fp8_size += fp8_size
            
            # Replace the parameter with a placeholder on meta device
            module, param_name = self._get_submodule_and_param_name(model, name)
            
            # Create a dummy parameter that will be replaced during forward
            dummy_param = nn.Parameter(
                torch.empty(param.shape, dtype=self.dtype, device='meta'),
                requires_grad=param.requires_grad
            )
            setattr(module, param_name, dummy_param)
        
        print(f"[FP8 Offload] Original size: {total_original_size / 1024**3:.2f} GB")
        print(f"[FP8 Offload] FP8 compressed size: {total_fp8_size / 1024**3:.2f} GB")
        print(f"[FP8 Offload] Compression ratio: {total_original_size / max(total_fp8_size, 1):.2f}x")
        
        return model
    
    def setup_module_hooks(self, module: nn.Module, module_name: str):
        """
        Setup forward/backward hooks for a specific module.
        
        This enables automatic weight loading before forward and
        offloading after backward.
        
        Args:
            module: The module to hook
            module_name: Name of the module (for matching weight wrappers)
        """
        # Find weight wrappers for this module
        module_wrappers = {}
        for name, wrapper in self.weight_wrappers.items():
            if name.startswith(module_name):
                # Get the local parameter name (within the module)
                local_name = name[len(module_name) + 1:] if module_name else name
                # Handle nested modules
                parts = local_name.split('.')
                if len(parts) == 1:
                    module_wrappers[parts[0]] = wrapper
        
        if not module_wrappers:
            return
        
        hook = FP8OffloadHook(module, module_wrappers, self.device, self.dtype)
        
        # Register hooks
        pre_hook = module.register_forward_pre_hook(hook.pre_forward)
        post_hook = module.register_forward_hook(hook.post_forward)
        
        self.hooks.append(pre_hook)
        self.hooks.append(post_hook)
        self.managed_modules[module_name] = module
    
    def load_weights_to_gpu(self, param_names: List[str] = None):
        """
        Manually load specified weights to GPU.
        
        Args:
            param_names: List of parameter names to load (if None, load all)
        """
        names = param_names or list(self.weight_wrappers.keys())
        for name in names:
            if name in self.weight_wrappers:
                self.weight_wrappers[name].to_gpu(self.device, self.dtype)
    
    def offload_weights_to_cpu(self, param_names: List[str] = None):
        """
        Manually offload specified weights to CPU.
        
        Args:
            param_names: List of parameter names to offload (if None, offload all)
        """
        names = param_names or list(self.weight_wrappers.keys())
        for name in names:
            if name in self.weight_wrappers:
                self.weight_wrappers[name].offload_to_cpu()
        torch.cuda.empty_cache()
    
    def update_weights(self, param_updates: Dict[str, torch.Tensor]):
        """
        Update FP8 weights from GPU tensors (after optimizer step).
        
        Args:
            param_updates: Dict mapping param names to updated GPU tensors
        """
        for name, gpu_weight in param_updates.items():
            if name in self.weight_wrappers:
                self.weight_wrappers[name].update_from_gpu(gpu_weight)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_gpu_weight(self, name: str) -> Optional[torch.Tensor]:
        """Get a weight tensor on GPU."""
        if name in self.weight_wrappers:
            return self.weight_wrappers[name].to_gpu(self.device, self.dtype)
        return None


class LayerWiseFP8Offloader:
    """
    Layer-wise FP8 offloader that loads/offloads weights per transformer block.
    
    This is more efficient for very large models as it only keeps one block's
    weights on GPU at a time.
    """
    
    def __init__(self, model: nn.Module, device: torch.device, 
                 dtype: torch.dtype = torch.bfloat16,
                 block_prefix: str = "blocks"):
        """
        Args:
            model: The model containing transformer blocks
            device: GPU device
            dtype: Computation dtype
            block_prefix: Prefix for transformer block names
        """
        self.model = model
        self.device = device
        self.dtype = dtype
        self.block_prefix = block_prefix
        
        # Dict mapping block_idx -> Dict[param_name -> FP8WeightWrapper]
        self.block_wrappers: Dict[int, Dict[str, FP8WeightWrapper]] = {}
        self.current_block_on_gpu: Optional[int] = None
        
        self._prepare_blocks()
    
    def _prepare_blocks(self):
        """Prepare FP8 wrappers for each transformer block."""
        print(f"[LayerWise FP8] Scanning for blocks with prefix '{self.block_prefix}'...")
        
        for name, module in self.model.named_modules():
            if self.block_prefix in name:
                # Extract block index
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == self.block_prefix.split('.')[-1]:
                        if i + 1 < len(parts) and parts[i + 1].isdigit():
                            block_idx = int(parts[i + 1])
                            if block_idx not in self.block_wrappers:
                                self.block_wrappers[block_idx] = {}
                            break
        
        # Now collect parameters for each block
        for name, param in self.model.named_parameters():
            if self.block_prefix not in name:
                continue
            
            # Extract block index
            parts = name.split('.')
            block_idx = None
            for i, part in enumerate(parts):
                if parts[i - 1] == self.block_prefix.split('.')[-1] if i > 0 else False:
                    if part.isdigit():
                        block_idx = int(part)
                        break
                elif part == self.block_prefix.split('.')[-1]:
                    if i + 1 < len(parts) and parts[i + 1].isdigit():
                        block_idx = int(parts[i + 1])
                        break
            
            if block_idx is not None and param.numel() > 1024:
                wrapper = FP8WeightWrapper(param, name)
                self.block_wrappers[block_idx][name] = wrapper
        
        num_blocks = len(self.block_wrappers)
        total_params = sum(len(w) for w in self.block_wrappers.values())
        print(f"[LayerWise FP8] Prepared {total_params} parameters across {num_blocks} blocks")
    
    def load_block(self, block_idx: int):
        """Load a specific block's weights to GPU."""
        if self.current_block_on_gpu == block_idx:
            return
        
        # Offload current block first
        if self.current_block_on_gpu is not None:
            self.offload_block(self.current_block_on_gpu)
        
        # Load new block
        if block_idx in self.block_wrappers:
            for name, wrapper in self.block_wrappers[block_idx].items():
                gpu_weight = wrapper.to_gpu(self.device, self.dtype)
                # Set the weight in the model
                self._set_param(name, gpu_weight)
        
        self.current_block_on_gpu = block_idx
    
    def offload_block(self, block_idx: int):
        """Offload a specific block's weights to CPU."""
        if block_idx in self.block_wrappers:
            for name, wrapper in self.block_wrappers[block_idx].items():
                wrapper.offload_to_cpu()
        
        if self.current_block_on_gpu == block_idx:
            self.current_block_on_gpu = None
        
        torch.cuda.empty_cache()
    
    def _set_param(self, full_name: str, gpu_weight: torch.Tensor):
        """Set a parameter in the model from its full name."""
        parts = full_name.split('.')
        module = self.model
        for part in parts[:-1]:
            module = getattr(module, part)
        param_name = parts[-1]
        
        original_param = getattr(module, param_name)
        setattr(module, param_name, nn.Parameter(gpu_weight, requires_grad=original_param.requires_grad))


def create_fp8_offload_optimizer_wrapper(optimizer, offload_manager: FP8CPUOffloadManager):
    """
    Create a wrapper around an optimizer that handles FP8 weight updates.
    
    This wrapper:
    1. Loads weights to GPU before optimizer step
    2. Updates the FP8 weights after step
    3. Offloads weights back to CPU
    
    Args:
        optimizer: The base optimizer
        offload_manager: The FP8CPUOffloadManager instance
        
    Returns:
        Wrapped optimizer
    """
    original_step = optimizer.step
    
    def wrapped_step(closure=None):
        # Perform the optimizer step
        result = original_step(closure)
        
        # Update FP8 weights from the optimizer's param groups
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    # Find the corresponding weight wrapper and update
                    for name, wrapper in offload_manager.weight_wrappers.items():
                        if wrapper._gpu_weight is not None and wrapper._gpu_weight.data_ptr() == param.data_ptr():
                            wrapper.update_from_gpu(param)
                            break
        
        return result
    
    optimizer.step = wrapped_step
    return optimizer
