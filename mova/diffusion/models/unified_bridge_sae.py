"""
Unified Sparse Autoencoder (SAE) for Dual-Tower Bridge Cross-Attention.

This module implements a unified SAE that learns sparse representations
of cross-modal interaction features from both audio-to-video and
video-to-audio conditioning pathways.

Architecture:
    a2v_output (3072) ──→ Proj_a2v ──┐
                                     ├──→ Unified Space (d_unified) ──→ SAE Encoder ──→ Sparse h ──→ SAE Decoder ──┐
    v2a_output (1536) ──→ Proj_v2a ─┘                                                                                   │
                                                                                                                                                        ├──→ UnProj_a2v ──→ (3072)
                                                                                                                                                        └──→ UnProj_v2a ──→ (1536)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TopKReLU(nn.Module):
    """Top-K ReLU activation: keeps only the top-k largest positive values."""
    
    def __init__(self, k: int):
        super().__init__()
        self.k = k
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only keep positive values
        x = F.relu(x)
        # Find top-k values
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)
        # Zero out everything except top-k
        result = torch.zeros_like(x)
        result.scatter_(-1, topk_indices, topk_values)
        return result
五

class UnifiedBridgeSAE(nn.Module):
    """
    Unified Sparse Autoencoder for dual-tower bridge cross-attention features.
    
    Projects features from both modalities into a shared latent space,
    learns sparse representations, and reconstructs back to original spaces.
    
    Args:
        visual_dim: Dimension of visual DiT hidden states (default: 3072)
        audio_dim: Dimension of audio DiT hidden states (default: 1536)
        unified_dim: Dimension of the unified latent space (default: 2048)
        sae_expansion: Expansion factor for SAE hidden dimension (default: 8)
        use_topk: Whether to use Top-K sparsity instead of L1 regularization (default: False)
        topk_k: Number of features to keep if use_topk=True (default: 32)
    """
    
    def __init__(
        self,
        visual_dim: int = 3072,
        audio_dim: int = 1536,
        unified_dim: int = 2048,
        sae_expansion: int = 8,
        use_topk: bool = False,
        topk_k: int = 32,
    ):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.unified_dim = unified_dim
        self.sae_hidden_dim = unified_dim * sae_expansion
        self.use_topk = use_topk
        self.topk_k = topk_k
        
        # Projection layers: modality-specific → unified space
        self.proj_a2v = nn.Linear(visual_dim, unified_dim)  # a2v conditioner output → unified
        self.proj_v2a = nn.Linear(audio_dim, unified_dim)   # v2a conditioner output → unified
        
        # SAE components in unified space
        self.sae_encoder = nn.Linear(unified_dim, self.sae_hidden_dim)
        self.sae_decoder = nn.Linear(self.sae_hidden_dim, unified_dim)
        
        # Unprojection layers: unified → modality-specific
        self.unproj_a2v = nn.Linear(unified_dim, visual_dim)
        self.unproj_v2a = nn.Linear(unified_dim, audio_dim)
        
        # Sparsity activation
        if use_topk:
            self.activation = TopKReLU(topk_k)
        else:
            self.activation = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection and SAE weights."""
        # Orthogonal initialization for better gradient flow
        nn.init.orthogonal_(self.proj_a2v.weight)
        nn.init.orthogonal_(self.proj_v2a.weight)
        nn.init.zeros_(self.proj_a2v.bias)
        nn.init.zeros_(self.proj_v2a.bias)
        
        # SAE encoder: normalized columns
        with torch.no_grad():
            self.sae_encoder.weight.data = F.normalize(self.sae_encoder.weight.data, p=2, dim=0)
        
        nn.init.zeros_(self.sae_encoder.bias)
        nn.init.zeros_(self.sae_decoder.bias)
        
        # Unprojection: initialize as pseudo-inverse of projection
        with torch.no_grad():
            self.unproj_a2v.weight.data = torch.linalg.pinv(self.proj_a2v.weight.data).T
            self.unproj_v2a.weight.data = torch.linalg.pinv(self.proj_v2a.weight.data).T
            self.unproj_a2v.bias.data.zero_()
            self.unproj_v2a.bias.data.zero_()
    
    def encode(self, x_unified: torch.Tensor) -> torch.Tensor:
        """Encode unified features to sparse latent codes."""
        h = self.sae_encoder(x_unified)
        h = self.activation(h)
        return h
    
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        """Decode sparse latent codes back to unified space."""
        return self.sae_decoder(h)
    
    def project(self, x: torch.Tensor, direction: str) -> torch.Tensor:
        """Project modality-specific features to unified space."""
        if direction == 'a2v':
            return self.proj_a2v(x)
        elif direction == 'v2a':
            return self.proj_v2a(x)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def unproject(self, h_unified: torch.Tensor, direction: str) -> torch.Tensor:
        """Unproject unified features back to modality-specific space."""
        if direction == 'a2v':
            return self.unproj_a2v(h_unified)
        elif direction == 'v2a':
            return self.unproj_v2a(h_unified)
        else:
            raise ValueError(f"Unknown direction: {direction}")
    
    def forward(
        self,
        x_a2v: Optional[torch.Tensor] = None,
        x_v2a: Optional[torch.Tensor] = None,
        direction: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for unified SAE.
        
        Args:
            x_a2v: a2v conditioner output [B, L, visual_dim]
            x_v2a: v2a conditioner output [B, L, audio_dim]
            direction: Process only one direction ('a2v' or 'v2a').
                      If None, process both.
        
        Returns:
            Dictionary containing:
                - 'reconstructed_a2v': Reconstructed a2v features (if input provided)
                - 'reconstructed_v2a': Reconstructed v2a features (if input provided)
                - 'sparse_code_a2v': Sparse codes for a2v (if input provided)
                - 'sparse_code_v2a': Sparse codes for v2a (if input provided)
                - 'unified_a2v': Unified space representation (if input provided)
                - 'unified_v2a': Unified space representation (if input provided)
        """
        result = {}
        
        if direction is None or direction == 'a2v':
            assert x_a2v is not None, "x_a2v is required for a2v direction"
            # Project → Encode → Decode → Unproject
            unified_a2v = self.proj_a2v(x_a2v)
            h_a2v = self.encode(unified_a2v)
            decoded_a2v = self.decode(h_a2v)
            reconstructed_a2v = self.unproj_a2v(decoded_a2v)
            
            result.update({
                'reconstructed_a2v': reconstructed_a2v,
                'sparse_code_a2v': h_a2v,
                'unified_a2v': unified_a2v,
            })
        
        if direction is None or direction == 'v2a':
            assert x_v2a is not None, "x_v2a is required for v2a direction"
            # Project → Encode → Decode → Unproject
            unified_v2a = self.proj_v2a(x_v2a)
            h_v2a = self.encode(unified_v2a)
            decoded_v2a = self.decode(h_v2a)
            reconstructed_v2a = self.unproj_v2a(decoded_v2a)
            
            result.update({
                'reconstructed_v2a': reconstructed_v2a,
                'sparse_code_v2a': h_v2a,
                'unified_v2a': unified_v2a,
            })
        
        return result
    
    def compute_loss(
        self,
        x_a2v: torch.Tensor,
        x_v2a: torch.Tensor,
        l1_lambda: float = 1e-3,
        recon_weight_a2v: float = 0.5,
        recon_weight_v2a: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute SAE training loss.
        
        Loss = recon_weight_a2v * MSE(reconstructed_a2v, x_a2v)
             + recon_weight_v2a * MSE(reconstructed_v2a, x_v2a)
             + l1_lambda * (||h_a2v||_1 + ||h_v2a||_1)
        
        Args:
            x_a2v: Original a2v conditioner output
            x_v2a: Original v2a conditioner output
            l1_lambda: L1 regularization strength for sparsity
            recon_weight_a2v: Weight for a2v reconstruction loss
            recon_weight_v2a: Weight for v2a reconstruction loss
        
        Returns:
            Dictionary containing loss components and total loss
        """
        # Forward pass
        output = self.forward(x_a2v=x_a2v, x_v2a=x_v2a)
        
        # Reconstruction losses
        recon_loss_a2v = F.mse_loss(output['reconstructed_a2v'], x_a2v)
        recon_loss_v2a = F.mse_loss(output['reconstructed_v2a'], x_v2a)
        
        # Weighted reconstruction loss
        recon_loss = recon_weight_a2v * recon_loss_a2v + recon_weight_v2a * recon_loss_v2a
        
        # Sparsity loss
        if self.use_topk:
            # Top-K already enforces sparsity, no L1 needed
            sparsity_loss = torch.tensor(0.0, device=x_a2v.device)
        else:
            h_a2v = output['sparse_code_a2v']
            h_v2a = output['sparse_code_v2a']
            sparsity_loss = h_a2v.abs().mean() + h_v2a.abs().mean()
        
        # Total loss
        total_loss = recon_loss + l1_lambda * sparsity_loss
        
        # Auxiliary metrics
        with torch.no_grad():
            # Fraction of dead neurons (never activated)
            h_a2v = output['sparse_code_a2v']
            h_v2a = output['sparse_code_v2a']
            dead_frac_a2v = (h_a2v.mean(dim=[0, 1]) < 1e-8).float().mean()
            dead_frac_v2a = (h_v2a.mean(dim=[0, 1]) < 1e-8).float().mean()
            
            # Variance explained
            var_explained_a2v = 1.0 - recon_loss_a2v / (x_a2v.var() + 1e-8)
            var_explained_v2a = 1.0 - recon_loss_v2a / (x_v2a.var() + 1e-8)
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'recon_loss_a2v': recon_loss_a2v,
            'recon_loss_v2a': recon_loss_v2a,
            'sparsity_loss': sparsity_loss,
            'var_explained_a2v': var_explained_a2v,
            'var_explained_v2a': var_explained_v2a,
            'dead_frac_a2v': dead_frac_a2v,
            'dead_frac_v2a': dead_frac_v2a,
        }
    
    def steer_features(
        self,
        x: torch.Tensor,
        direction: str,
        intervention_mask: torch.Tensor,
        intervention_strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Intervene on sparse codes to control output features.
        
        Args:
            x: Input conditioner output (a2v or v2a)
            direction: 'a2v' or 'v2a'
            intervention_mask: Binary mask indicating which features to modify [sae_hidden_dim]
            intervention_strength: Strength of intervention (0.0 = no change, 1.0 = full suppression)
        
        Returns:
            Modified conditioner output after intervention
        """
        # Project to unified space
        unified = self.project(x, direction)
        
        # Encode to sparse code
        h = self.encode(unified)
        
        # Apply intervention (suppress selected features)
        h_modified = h * (1.0 - intervention_strength * intervention_mask.unsqueeze(0).unsqueeze(0))
        
        # Decode and unproject
        decoded = self.decode(h_modified)
        reconstructed = self.unproject(decoded, direction)
        
        return reconstructed


class SAEActivationCollector:
    """
    Hook-based collector for gathering SAE training data from the bridge.
    
    Usage:
        collector = SAEActivationCollector(bridge)
        # Run inference/training
        collector.collect_activations(model_inputs)
        # Get collected data
        activations = collector.get_activations()
    """
    
    def __init__(self, bridge: nn.Module):
        self.bridge = bridge
        self.activations_a2v = []
        self.activations_v2a = []
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all conditioners."""
        # Hook a2v conditioners
        for name, conditioner in self.bridge.audio_to_video_conditioners.items():
            hook = conditioner.inner.register_forward_hook(
                self._make_hook('a2v', name)
            )
            self.hooks.append(hook)
        
        # Hook v2a conditioners
        for name, conditioner in self.bridge.video_to_audio_conditioners.items():
            hook = conditioner.inner.register_forward_hook(
                self._make_hook('v2a', name)
            )
            self.hooks.append(hook)
    
    def _make_hook(self, direction: str, layer_name: str):
        """Create a forward hook for collecting activations."""
        def hook_fn(module, args, kwargs, output):
            # output is the cross-attention output
            self.activations_a2v.append(output.detach().cpu()) if direction == 'a2v' else None
            self.activations_v2a.append(output.detach().cpu()) if direction == 'v2a' else None
        return hook_fn
    
    def get_activations(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all collected activations."""
        if self.activations_a2v:
            a2v = torch.cat(self.activations_a2v, dim=0)
        else:
            a2v = torch.empty(0)
        
        if self.activations_v2a:
            v2a = torch.cat(self.activations_v2a, dim=0)
        else:
            v2a = torch.empty(0)
        
        return a2v, v2a
    
    def clear(self):
        """Clear collected activations."""
        self.activations_a2v.clear()
        self.activations_v2a.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
