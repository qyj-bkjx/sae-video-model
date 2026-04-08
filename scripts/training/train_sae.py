"""
Training script for Unified Bridge SAE.

This script trains the unified SAE on collected bridge cross-attention activations
from the dual-tower MOVA model.

Usage:
    # Phase 1: Collect activations
    python scripts/training/train_sae.py --phase collect \
        --checkpoint_path /path/to/mova_checkpoint \
        --data_dir /path/to/training/data \
        --num_samples 10000 \
        --save_path /path/to/save/activations
    
    # Phase 2: Train SAE
    python scripts/training/train_sae.py --phase train \
        --activation_path /path/to/saved/activations \
        --unified_dim 2048 \
        --sae_expansion 8 \
        --l1_lambda 0.001 \
        --epochs 50 \
        --batch_size 256 \
        --lr 1e-3 \
        --save_path /path/to/save/sae
    
    # Phase 3: Evaluate SAE
    python scripts/training/train_sae.py --phase eval \
        --sae_path /path/to/trained/sae \
        --activation_path /path/to/test/activations
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mova.diffusion.models.unified_bridge_sae import (
    UnifiedBridgeSAE,
    SAEActivationCollector,
)


def collect_activations(
    checkpoint_path: str,
    data_dir: str,
    num_samples: int,
    save_path: str,
    device: str = "cuda",
    batch_size: int = 1,
):
    """
    Phase 1: Collect bridge cross-attention activations.
    
    This runs inference on the MOVA model and collects the outputs
    from all bridge cross-attention conditioners.
    """
    print("=" * 60)
    print("Phase 1: Collecting Bridge Activations")
    print("=" * 60)
    
    # Load MOVA pipeline
    from mova.diffusion.pipelines.mova_train import MOVAPipeline
    
    # Placeholder: actual loading depends on your checkpoint format
    # This is a simplified example - adjust based on your actual setup
    print(f"Loading model from {checkpoint_path}...")
    
    # You'll need to adapt this based on your actual checkpoint format
    # Typically involves loading video_dit, audio_dit, and dual_tower_bridge
    print("[NOTE] Adjust model loading based on your actual checkpoint format")
    print("Expected components:")
    print("  - video_dit (visual DiT)")
    print("  - audio_dit (audio DiT)")  
    print("  - dual_tower_bridge (bridge cross-attention)")
    
    # For now, we simulate the collection process
    # In practice, you'd hook into the actual pipeline
    print("\nTo collect real activations, you need to:")
    print("1. Load the MOVA pipeline with your checkpoint")
    print("2. Register SAEActivationCollector hooks on dual_tower_bridge")
    print("3. Run inference on your training data")
    print("4. Save collected activations")
    
    # Simulated activation collection for demonstration
    print("\nSimulating activation collection...")
    visual_dim = 3072
    audio_dim = 1536
    
    # Generate synthetic activations (replace with real collection)
    activations_a2v = torch.randn(min(num_samples, 1000), 64, visual_dim)
    activations_v2a = torch.randn(min(num_samples, 1000), 64, audio_dim)
    
    # Save activations
    os.makedirs(save_path, exist_ok=True)
    torch.save(activations_a2v, os.path.join(save_path, "activations_a2v.pt"))
    torch.save(activations_v2a, os.path.join(save_path, "activations_v2a.pt"))
    
    print(f"\nSaved activations to {save_path}/")
    print(f"  a2v shape: {activations_a2v.shape}")
    print(f"  v2a shape: {activations_v2a.shape}")
    
    return activations_a2v, activations_v2a


class SAEDataset(TensorDataset):
    """Dataset for SAE training with paired a2v and v2a activations."""
    
    def __init__(self, activations_a2v: torch.Tensor, activations_v2a: torch.Tensor):
        # Ensure same number of samples
        min_len = min(len(activations_a2v), len(activations_v2a))
        self.a2v = activations_a2v[:min_len]
        self.v2a = activations_v2a[:min_len]
    
    def __getitem__(self, idx):
        return self.a2v[idx], self.v2a[idx]


def train_sae(
    activation_path: str,
    save_path: str,
    unified_dim: int = 2048,
    sae_expansion: int = 8,
    use_topk: bool = False,
    topk_k: int = 32,
    l1_lambda: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    warmup_steps: int = 100,
    device: str = "cuda",
):
    """
    Phase 2: Train the unified SAE.
    """
    print("=" * 60)
    print("Phase 2: Training Unified Bridge SAE")
    print("=" * 60)
    
    # Load activations
    print(f"Loading activations from {activation_path}...")
    activations_a2v = torch.load(
        os.path.join(activation_path, "activations_a2v.pt"),
        map_location="cpu"
    )
    activations_v2a = torch.load(
        os.path.join(activation_path, "activations_v2a.pt"),
        map_location="cpu"
    )
    
    print(f"Loaded a2v activations: {activations_a2v.shape}")
    print(f"Loaded v2a activations: {activations_v2a.shape}")
    
    # Flatten sequence dimension for training
    # Shape: [N, L, D] → [N*L, D]
    visual_dim = activations_a2v.shape[-1]
    audio_dim = activations_v2a.shape[-1]
    
    # Subsample if too large
    max_samples = 500000  # Limit for memory
    a2v_flat = activations_a2v.view(-1, visual_dim)[:max_samples]
    v2a_flat = activations_v2a.view(-1, audio_dim)[:max_samples]
    
    print(f"Flattened a2v: {a2v_flat.shape}")
    print(f"Flattened v2a: {v2a_flat.shape}")
    
    # Create dataset and dataloader
    dataset = SAEDataset(a2v_flat, v2a_flat)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize SAE
    sae = UnifiedBridgeSAE(
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        unified_dim=unified_dim,
        sae_expansion=sae_expansion,
        use_topk=use_topk,
        topk_k=topk_k,
    ).to(device)
    
    print(f"\nSAE Architecture:")
    print(f"  Visual dim: {visual_dim}")
    print(f"  Audio dim: {audio_dim}")
    print(f"  Unified dim: {unified_dim}")
    print(f"  SAE hidden dim: {sae.sae_hidden_dim}")
    print(f"  Use TopK: {use_topk}")
    print(f"  L1 lambda: {l1_lambda}")
    
    # Optimizer
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    history = {
        'total_loss': [],
        'recon_loss': [],
        'recon_loss_a2v': [],
        'recon_loss_v2a': [],
        'sparsity_loss': [],
        'var_explained_a2v': [],
        'var_explained_v2a': [],
    }
    
    global_step = 0
    for epoch in range(epochs):
        sae.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_a2v, batch_v2a in pbar:
            batch_a2v = batch_a2v.to(device)
            batch_v2a = batch_v2a.to(device)
            
            optimizer.zero_grad()
            
            # Compute loss
            loss_dict = sae.compute_loss(
                batch_a2v, batch_v2a,
                l1_lambda=l1_lambda
            )
            
            loss = loss_dict['total_loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            epoch_losses.append(loss_dict)
            global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'recon_a2v': f"{loss_dict['recon_loss_a2v'].item():.4f}",
                'recon_v2a': f"{loss_dict['recon_loss_v2a'].item():.4f}",
                'var_a2v': f"{loss_dict['var_explained_a2v'].item():.4f}",
                'var_v2a': f"{loss_dict['var_explained_v2a'].item():.4f}",
            })
        
        # Epoch summary
        avg_losses = {
            k: torch.stack([d[k] for d in epoch_losses]).mean().item()
            for k in history.keys()
        }
        
        for k in history:
            history[k].append(avg_losses[k])
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Total Loss: {avg_losses['total_loss']:.4f}")
        print(f"  Recon Loss (a2v): {avg_losses['recon_loss_a2v']:.4f}")
        print(f"  Recon Loss (v2a): {avg_losses['recon_loss_v2a']:.4f}")
        print(f"  Var Explained (a2v): {avg_losses['var_explained_a2v']:.4f}")
        print(f"  Var Explained (v2a): {avg_losses['var_explained_v2a']:.4f}")
        print(f"  Sparsity Loss: {avg_losses['sparsity_loss']:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'sae_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': {
                    'visual_dim': visual_dim,
                    'audio_dim': audio_dim,
                    'unified_dim': unified_dim,
                    'sae_expansion': sae_expansion,
                    'use_topk': use_topk,
                    'topk_k': topk_k,
                    'l1_lambda': l1_lambda,
                }
            }
            
            os.makedirs(save_path, exist_ok=True)
            ckpt_path = os.path.join(save_path, f"sae_checkpoint_epoch_{epoch+1}.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"  Saved checkpoint to {ckpt_path}")
    
    # Final save
    final_checkpoint = {
        'epoch': epochs,
        'sae_state_dict': sae.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': {
            'visual_dim': visual_dim,
            'audio_dim': audio_dim,
            'unified_dim': unified_dim,
            'sae_expansion': sae_expansion,
            'use_topk': use_topk,
            'topk_k': topk_k,
            'l1_lambda': l1_lambda,
        }
    }
    
    final_path = os.path.join(save_path, "sae_final.pt")
    torch.save(final_checkpoint, final_path)
    
    # Save training history plot
    plot_training_history(history, save_path)
    
    print(f"\nTraining complete! Final SAE saved to {final_path}")
    
    return sae, history


def plot_training_history(history: dict, save_path: str):
    """Plot and save training history."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Total loss
    axes[0, 0].plot(history['total_loss'])
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Reconstruction losses
    axes[0, 1].plot(history['recon_loss_a2v'], label='a2v')
    axes[0, 1].plot(history['recon_loss_v2a'], label='v2a')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Variance explained
    axes[1, 0].plot(history['var_explained_a2v'], label='a2v')
    axes[1, 0].plot(history['var_explained_v2a'], label='v2a')
    axes[1, 0].set_title('Variance Explained')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Var Explained')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Sparsity loss
    axes[1, 1].plot(history['sparsity_loss'])
    axes[1, 1].set_title('Sparsity Loss (L1)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('L1 Norm')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, "training_history.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training history plot saved to {plot_path}")


def evaluate_sae(
    sae_path: str,
    activation_path: str,
    device: str = "cuda",
):
    """
    Phase 3: Evaluate trained SAE.
    """
    print("=" * 60)
    print("Phase 3: Evaluating Trained SAE")
    print("=" * 60)
    
    # Load SAE
    checkpoint = torch.load(sae_path, map_location="cpu")
    config = checkpoint['config']
    
    sae = UnifiedBridgeSAE(
        visual_dim=config['visual_dim'],
        audio_dim=config['audio_dim'],
        unified_dim=config['unified_dim'],
        sae_expansion=config['sae_expansion'],
        use_topk=config.get('use_topk', False),
        topk_k=config.get('topk_k', 32),
    )
    sae.load_state_dict(checkpoint['sae_state_dict'])
    sae = sae.to(device)
    sae.eval()
    
    print(f"Loaded SAE from {sae_path}")
    print(f"Config: {config}")
    
    # Load test activations
    activations_a2v = torch.load(
        os.path.join(activation_path, "activations_a2v.pt"),
        map_location="cpu"
    )
    activations_v2a = torch.load(
        os.path.join(activation_path, "activations_v2a.pt"),
        map_location="cpu"
    )
    
    # Flatten
    visual_dim = activations_a2v.shape[-1]
    audio_dim = activations_v2a.shape[-1]
    a2v_flat = activations_a2v.view(-1, visual_dim).to(device)
    v2a_flat = activations_v2a.view(-1, audio_dim).to(device)
    
    # Evaluate
    with torch.no_grad():
        loss_dict = sae.compute_loss(a2v_flat[:10000], v2a_flat[:10000])
    
    print("\nEvaluation Results:")
    print(f"  Total Loss: {loss_dict['total_loss'].item():.4f}")
    print(f"  Recon Loss (a2v): {loss_dict['recon_loss_a2v'].item():.4f}")
    print(f"  Recon Loss (v2a): {loss_dict['recon_loss_v2a'].item():.4f}")
    print(f"  Var Explained (a2v): {loss_dict['var_explained_a2v'].item():.4f}")
    print(f"  Var Explained (v2a): {loss_dict['var_explained_v2a'].item():.4f}")
    print(f"  Sparsity Loss: {loss_dict['sparsity_loss'].item():.4f}")
    
    # Analyze feature importance
    print("\nAnalyzing sparse code statistics...")
    with torch.no_grad():
        output = sae.forward(x_a2v=a2v_flat[:1000], x_v2a=v2a_flat[:1000])
        
        h_a2v = output['sparse_code_a2v']
        h_v2a = output['sparse_code_v2a']
        
        # Feature activity
        activity_a2v = (h_a2v.abs().mean(dim=[0, 1]) > 1e-6).sum().item()
        activity_v2a = (h_v2a.abs().mean(dim=[0, 1]) > 1e-6).sum().item()
        
        print(f"  Active features (a2v): {activity_a2v}/{sae.sae_hidden_dim}")
        print(f"  Active features (v2a): {activity_v2a}/{sae.sae_hidden_dim}")
        
        # Top features by mean activation
        top_features_a2v = h_a2v.abs().mean(dim=[0, 1]).topk(10)
        print(f"  Top 10 a2v features: {top_features_a2v.indices.tolist()}")
        print(f"  Top 10 a2v values: {top_features_a2v.values.tolist()}")
    
    return sae


def main():
    parser = argparse.ArgumentParser(description="Train Unified Bridge SAE")
    parser.add_argument("--phase", type=str, required=True, 
                       choices=["collect", "train", "eval"],
                       help="Training phase")
    
    # Collection args
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to MOVA checkpoint")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Path to training data")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of samples to collect")
    
    # Training args
    parser.add_argument("--activation_path", type=str, default=None,
                       help="Path to collected activations")
    parser.add_argument("--unified_dim", type=int, default=2048,
                       help="Dimension of unified latent space")
    parser.add_argument("--sae_expansion", type=int, default=8,
                       help="SAE expansion factor")
    parser.add_argument("--use_topk", action="store_true",
                       help="Use Top-K sparsity instead of L1")
    parser.add_argument("--topk_k", type=int, default=32,
                       help="K for Top-K sparsity")
    parser.add_argument("--l1_lambda", type=float, default=1e-3,
                       help="L1 regularization strength")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=256,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    
    # Common args
    parser.add_argument("--save_path", type=str, required=True,
                       help="Path to save outputs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use")
    
    args = parser.parse_args()
    
    if args.phase == "collect":
        collect_activations(
            checkpoint_path=args.checkpoint_path,
            data_dir=args.data_dir,
            num_samples=args.num_samples,
            save_path=args.save_path,
            device=args.device,
        )
    elif args.phase == "train":
        train_sae(
            activation_path=args.activation_path,
            save_path=args.save_path,
            unified_dim=args.unified_dim,
            sae_expansion=args.sae_expansion,
            use_topk=args.use_topk,
            topk_k=args.topk_k,
            l1_lambda=args.l1_lambda,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    elif args.phase == "eval":
        evaluate_sae(
            sae_path=os.path.join(args.save_path, "sae_final.pt"),
            activation_path=args.activation_path,
            device=args.device,
        )


if __name__ == "__main__":
    main()
