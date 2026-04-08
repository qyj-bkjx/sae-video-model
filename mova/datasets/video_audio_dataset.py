import json
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchcodec.decoders import AudioDecoder, VideoDecoder
from torchvision.transforms import InterpolationMode

from mova.registry import DATASETS



@DATASETS.register_module()
class VideoAudioDataset(Dataset):
    """
    Video-Audio Joint Training Dataset
    
    Data format requirements:
    data_root/
    ├── metadata.json  # [{"video_path": "xxx.mp4", "caption": "..."}]
    ├── videos/
    """
    
    def __init__(
        self,
        data_root: str,
        metadata_file: str = "metadata.json",
        num_frames: int = 49,
        height: int = 480,
        width: int = 720,
        sample_rate: int = 48000,
        video_fps: float = 24.0,
        transform=None,
        audio_transform=None,
    ):
        super().__init__()
        self.data_root = data_root
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.sample_rate = sample_rate
        self.video_fps = video_fps
        self.transform = transform
        self.audio_transform = audio_transform
        
        metadata_path = os.path.join(data_root, metadata_file)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        print(f"Loaded {len(self.metadata)} samples from {metadata_path}")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict:
        item = self.metadata[idx]
        
        video_path = os.path.join(self.data_root, item["video_path"])
        video_frames = self._load_video(video_path)
        
        audio = self._extract_audio_from_video(video_path)
        
        caption = item.get("caption", "")
        
        if self.transform is not None:
            video_frames = self.transform(video_frames)
        
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)
            
        first_frame = video_frames[0]
        
        return {
            "video": video_frames,          # [T, C, H, W]
            "audio": audio,                 # [1, T_audio]
            "first_frame": first_frame,     # [C, H, W]
            "caption": caption,
            "idx": idx,
        }
    
    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load video and sample frames"""
        decoder = VideoDecoder(video_path, device="cpu")

        total_frames = decoder.metadata.num_frames
        target_frames = min(total_frames, self.num_frames)

        indices = list(range(target_frames))

        frame_batch = decoder.get_frames_at(indices=indices)
        frames = frame_batch.data
        
        in_h, in_w = int(frames.shape[-2]), int(frames.shape[-1])
        target_ratio = float(self.width) / float(self.height)
        in_ratio = float(in_w) / float(in_h) if in_h > 0 else target_ratio

        if in_ratio > target_ratio:
            crop_h = in_h
            crop_w = max(1, int(round(in_h * target_ratio)))
        else:
            crop_w = in_w
            crop_h = max(1, int(round(in_w / target_ratio)))

        frames = TF.center_crop(frames, [crop_h, crop_w])
        frames = TF.resize(
            frames,
            [self.height, self.width],
            interpolation=InterpolationMode.BILINEAR,
        )

        frames = frames.float() / 255.0
        
        frames = frames * 2 - 1
        
        return frames
    
    def _extract_audio_from_video(self, video_path: str) -> torch.Tensor:
        """
        Extract audio directly from video file
        
        Use torchcodec to decode audio stream from video container
        """
        duration_s = float(self.num_frames) / float(self.video_fps)
        audio_decoder = AudioDecoder(video_path, sample_rate=self.sample_rate, num_channels=1)
        audio_samples = audio_decoder.get_samples_played_in_range(
            start_seconds=0.0, stop_seconds=duration_s
        )

        waveform = audio_samples.data

        pts_seconds = audio_samples.pts_seconds
        if pts_seconds > 0.0:
            left_pad = int(round(pts_seconds * self.sample_rate))
            waveform = torch.nn.functional.pad(waveform, (left_pad, 0))

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        assert waveform.ndim == 2, f"Waveform shape mismatch: expected [1, T], got {waveform.shape} path={video_path}"

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        target_samples = int(self.sample_rate * self.num_frames / self.video_fps)
        
        if waveform.shape[1] >= target_samples:
            waveform = waveform[:, :target_samples]
        else:
            padding = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        return waveform


def collate_fn(batch):
    """Custom collate function."""
    videos = torch.stack([item["video"] for item in batch])
    audios = torch.stack([item["audio"] for item in batch])
    first_frames = torch.stack([item["first_frame"] for item in batch])
    captions = [item["caption"] for item in batch]
    idxs = [item["idx"] for item in batch]
    return {
        "video": videos,              # [B, T, C, H, W]
        "audio": audios,              # [B, 1, T_audio]
        "first_frame": first_frames,  # [B, C, H, W]
        "caption": captions,
        "idx": idxs,
    }

