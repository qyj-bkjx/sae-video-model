#!/usr/bin/env python3
"""
MOVA LoRA Inference Script
"""

import os
import argparse
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from PIL import Image

from mova.diffusion.pipelines import MOVALoRA
from mova.utils.data import save_video_with_audio
from mova.datasets.transforms.custom import crop_and_resize


DEFAULT_PROMPT = (
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
    "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
    "Blurred beach scenery forms the background featuring crystal-clear waters, distant "
    "green hills, and a blue sky dotted with white clouds."
)

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指"
)


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def parse_args():
    parser = argparse.ArgumentParser(description="MOVA LoRA Inference")
    
    # Model paths
    parser.add_argument("--base_model", type=str, required=True, 
                        help="Base model path (e.g., /path/to/A14B-360p-wsd-105000)")
    parser.add_argument("--lora_path", type=str, required=True,
                        help="LoRA checkpoint path (contains lora_weights.pt)")
    parser.add_argument("--lora_alpha", type=float, default=None,
                        help="Override LoRA alpha scale")
    parser.add_argument("--lora_scale", type=float, default=1.0,
                        help="LoRA effect strength (0.0-1.0)")
    parser.add_argument("--merge_lora", action="store_true",
                        help="Merge LoRA weights into base model (faster inference)")
    
    # Input and output
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT,
                        help="Text prompt for generation")
    parser.add_argument("--negative_prompt", type=str, default=NEGATIVE_PROMPT,
                        help="Negative prompt")
    parser.add_argument("--ref_path", type=str, default="./data/samples/i2v.jpg",
                        help="Reference image path")
    parser.add_argument("--output_path", type=str, default="./output_lora.mp4",
                        help="Output video path")
    
    # Generation parameters
    parser.add_argument("--num_frames", type=int, default=193,
                        help="Number of frames to generate")
    parser.add_argument("--fps", type=float, default=24.0,
                        help="Video FPS")
    parser.add_argument("--height", type=int, default=352,
                        help="Video height")
    parser.add_argument("--width", type=int, default=640,
                        help="Video width")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--sigma_shift", type=float, default=5.0,
                        help="Sigma shift for scheduler")
    
    # Distributed parameters
    parser.add_argument("--cp_size", type=int, default=1,
                        help="Context Parallel size")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.ref_path):
        raise FileNotFoundError(f"Reference image not found: {args.ref_path}")
    if not os.path.exists(args.lora_path):
        raise FileNotFoundError(f"LoRA path not found: {args.lora_path}")
    
    # --------------------------------------------------
    # Initialize distributed environment
    # --------------------------------------------------
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", device_id=local_rank)
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        torch.cuda.set_device(0)
    
    device = "cuda"
    torch_dtype = torch.bfloat16
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # --------------------------------------------------
    # Load model + LoRA
    # --------------------------------------------------
    if is_main_process():
        print(f"Loading base model from {args.base_model}")
        print(f"Loading LoRA from {args.lora_path}")
    
    pipeline = MOVALoRA.from_pretrained_with_lora(
        pretrained_path=args.base_model,
        lora_path=args.lora_path,
        device=device,
        torch_dtype=torch_dtype,
        lora_alpha=args.lora_alpha,
    )

    
    # Set LoRA strength
    if args.lora_scale != 1.0:
        pipeline.set_lora_scale(args.lora_scale)
        if is_main_process():
            print(f"Set LoRA scale to {args.lora_scale}")
    
    # Optional: merge LoRA weights
    if args.merge_lora:
        pipeline.merge_lora_weights()
        if is_main_process():
            print("LoRA weights merged into base model")
    
    # --------------------------------------------------
    # Context Parallel settings
    # --------------------------------------------------
    cp_mesh = None
    if args.cp_size > 1 and dist.is_initialized():
        from yunchang import set_seq_parallel_pg
        
        dp_size = world_size // args.cp_size
        mesh = DeviceMesh(
            "cuda",
            torch.arange(world_size).view(dp_size, args.cp_size),
            mesh_dim_names=("dp", "cp"),
        )
        cp_mesh = mesh["cp"]
        
        MAX_ULYSSES_DEGREE = 4
        ulysses_degree = min(args.cp_size, MAX_ULYSSES_DEGREE)
        ring_degree = args.cp_size // ulysses_degree
        
        set_seq_parallel_pg(
            sp_ulysses_degree=ulysses_degree,
            sp_ring_degree=ring_degree,
            rank=rank,
            world_size=world_size,
            use_ulysses_low=True,
        )
        
        replaced = pipeline.replace_attention()
        if is_main_process():
            print(f"[CP] Enabled, cp_size={args.cp_size}, replaced {replaced} attention modules")
    
    # --------------------------------------------------
    # Prepare input
    # --------------------------------------------------
    img = Image.open(args.ref_path).convert("RGB")
    ref_img = crop_and_resize(img, height=args.height, width=args.width)
    
    if is_main_process():
        print(f"Reference image: {args.ref_path}")
        print(f"Prompt: {args.prompt[:100]}...")
        print(f"Output: {args.output_path}")
        print(f"Resolution: {args.width}x{args.height}, Frames: {args.num_frames}")
    
    # --------------------------------------------------
    # Inference
    # --------------------------------------------------
    if is_main_process():
        print("Starting inference...")
    
    # pipeline.eval()
    
    with torch.no_grad(), torch.autocast("cuda", dtype=torch_dtype):
        video, audio = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_frames=args.num_frames,
            image=ref_img,
            height=args.height,
            width=args.width,
            video_fps=args.fps,
            num_inference_steps=args.num_inference_steps,
            sigma_shift=args.sigma_shift,
            cfg_scale=args.cfg_scale,
            seed=args.seed,
            cp_mesh=cp_mesh,
        )
    
    # --------------------------------------------------
    # Save results (only main process)
    # --------------------------------------------------
    if is_main_process():
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        audio_save = audio[0].cpu().squeeze()
        
        save_video_with_audio(
            video[0],
            audio_save,
            args.output_path,
            fps=args.fps,
            sample_rate=pipeline.audio_sample_rate,
            quality=5,
        )
        
        print(f"Saved to {args.output_path}")
    
    # --------------------------------------------------
    # Clean up
    # --------------------------------------------------
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
