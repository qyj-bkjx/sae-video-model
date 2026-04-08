#!/usr/bin/env python3
"""
SGLang API Client for Video Generation

Provides a Python client for interacting with SGLang video generation API.
Uses requests library for direct API communication.
"""

import os
import requests
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from config import SGLANG_SERVERS


@dataclass
class VideoTask:
    """Represents a video generation task."""
    id: str
    status: str
    progress: Optional[int] = None
    created_at: Optional[int] = None
    completed_at: Optional[int] = None
    download_url: Optional[str] = None
    video_url: Optional[str] = None  # Alias for download_url, some APIs use this field name
    file_path: Optional[str] = None  # Local file path on server
    inference_time_s: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    error: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoTask':
        """Create VideoTask from API response dictionary."""
        # Handle both video_url and download_url fields
        download_url = data.get('download_url') or data.get('video_url')
        return cls(
            id=data.get('id', ''),
            status=data.get('status', 'unknown'),
            progress=data.get('progress'),
            created_at=data.get('created_at'),
            completed_at=data.get('completed_at'),
            download_url=download_url,
            video_url=download_url,  # Set both for compatibility
            file_path=data.get('file_path'),  # Local file path on server
            inference_time_s=data.get('inference_time_s'),
            peak_memory_mb=data.get('peak_memory_mb'),
            error=data.get('error')
        )


class SGLangClient:
    """Client for SGLang video generation API."""
    
    def __init__(self, server_key: str):
        """
        Initialize SGLang client.
        
        Args:
            server_key: Key for server configuration (e.g., "mova-360p")
        """
        if server_key not in SGLANG_SERVERS:
            raise ValueError(f"Unknown server key: {server_key}. Available: {list(SGLANG_SERVERS.keys())}")
        
        self.server_config = SGLANG_SERVERS[server_key]
        self.base_url = self.server_config['base_url'].rstrip('/')
    
    def _parse_task_response(self, result) -> VideoTask:
        """Parse API response (list or dict) into VideoTask."""
        if isinstance(result, list) and len(result) > 0:
            task_data = result[0]
        elif isinstance(result, dict):
            task_data = result
        else:
            raise ValueError(f"Unexpected API response format: {result}")
        return VideoTask.from_dict(task_data)
    
    def _normalize_download_url(self, url: str) -> str:
        """Remove double slashes in path (e.g. //v1 -> /v1)."""
        if '://' not in url:
            return url
        protocol, path = url.split('://', 1)
        path = '/' + path.lstrip('/')
        return f"{protocol}://{path}"
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        return {}
    
    def submit_video_task(
        self,
        prompt: str,
        image_path: str,
        size: str = "640x360",
        num_frames: int = 193,
        fps: int = 24,
        seed: Optional[int] = None,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 50
    ) -> VideoTask:
        """
        Submit a video generation task.
        
        Args:
            prompt: Text prompt describing the video
            image_path: Path to the first frame image
            size: Video size (e.g., "640x360" for 360p landscape)
            num_frames: Number of frames to generate
            fps: Frames per second
            seed: Random seed for reproducibility (None or 0 means random)
            guidance_scale: Classifier-free guidance scale
            num_inference_steps: Number of denoising steps
            
        Returns:
            VideoTask object with task ID and initial status
            
        Raises:
            FileNotFoundError: If image_path doesn't exist
            requests.HTTPError: If API request fails
        """
        # Verify image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Prepare multipart form data
        url = f"{self.base_url}/v1/videos"
        
        # Read image file
        with open(image_path, 'rb') as f:
            files = {
                'input_reference': (os.path.basename(image_path), f, 'image/png')
            }
            
            # Only pass num_frames, not seconds, to let server use num_frames directly
            # According to SGLang API: if num_frames is provided, it takes priority
            data = {
                'prompt': prompt,
                'size': size,
                'num_frames': str(num_frames),  # Primary parameter - should take priority
                'fps': str(fps),
                'guidance_scale': str(guidance_scale),
                'num_inference_steps': str(num_inference_steps)
            }
            
            # Generate random seed if seed is None or 0
            if seed is None or seed == 0:
                import random
                seed = random.randint(1, 2**31 - 1)  # Generate random seed
            data['seed'] = str(seed)
            
            response = requests.post(
                url,
                headers=self._get_headers(),
                data=data,
                files=files,
                timeout=60
            )
        
        response.raise_for_status()
        return self._parse_task_response(response.json())
    
    def get_task_status(self, task_id: str) -> VideoTask:
        """
        Get the status of a video generation task.
        
        Args:
            task_id: Task ID returned from submit_video_task
            
        Returns:
            VideoTask object with current status
            
        Raises:
            requests.HTTPError: If API request fails
        """
        url = f"{self.base_url}/v1/videos/{task_id}"
        response = requests.get(url, headers=self._get_headers(), timeout=30)
        response.raise_for_status()
        return self._parse_task_response(response.json())
    
    def get_download_url(self, task_id: str) -> str:
        """
        Get the download URL for a completed video.
        
        Args:
            task_id: Task ID
            
        Returns:
            Download URL string
            
        Raises:
            requests.HTTPError: If API request fails
        """
        task = self.get_task_status(task_id)
        
        if task.download_url:
            return self._normalize_download_url(task.download_url)
        return f"{self.base_url}/v1/videos/{task_id}/download"
    
    def download_video(self, task_id: str, save_path: str) -> str:
        """
        Download a completed video to local path.
        First tries to copy from local file_path if available, otherwise downloads via HTTP.
        
        Args:
            task_id: Task ID
            save_path: Local path to save the video file
            
        Returns:
            Path to the downloaded video file
            
        Raises:
            requests.HTTPError: If download fails
            FileNotFoundError: If save directory doesn't exist or source file doesn't exist
        """
        # Ensure save directory exists
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Get task status to check for file_path
        task = self.get_task_status(task_id)
        
        # If file_path is available and file exists, copy from local filesystem
        if task.file_path and Path(task.file_path).exists():
            try:
                import shutil
                print(f"Copying video from local file: {task.file_path}")
                shutil.copy2(task.file_path, save_path)
                
                # Verify file was copied
                if not Path(save_path).exists() or Path(save_path).stat().st_size == 0:
                    raise IOError(f"Copied file is empty or doesn't exist: {save_path}")
                
                file_size = Path(save_path).stat().st_size
                print(f"Successfully copied video to {save_path} ({file_size} bytes)")
                return save_path
                
            except Exception as copy_error:
                print(f"Failed to copy from local file_path: {copy_error}, trying HTTP download...")
                # Fall through to HTTP download
        
        # If no file_path or copy failed, try HTTP download
        download_url = self.get_download_url(task_id)
        
        try:
            response = requests.get(
                download_url,
                headers=self._get_headers(),
                stream=True,
                timeout=300  # 5 minutes for large videos
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '')
            if 'video' not in content_type and 'octet-stream' not in content_type:
                print(f"Warning: Unexpected content type: {content_type}")
            
            # Save to file
            total_size = 0
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)
            
            # Verify file was written
            if not Path(save_path).exists() or Path(save_path).stat().st_size == 0:
                raise IOError(f"Downloaded file is empty or doesn't exist: {save_path}")
            
            print(f"Successfully downloaded video to {save_path} ({total_size} bytes)")
            return save_path
            
        except requests.exceptions.RequestException as e:
            error_msg = f"HTTP error downloading video: {e}"
            if hasattr(e, 'response') and e.response is not None:
                error_msg += f" (Status: {e.response.status_code}, Response: {e.response.text[:200]})"
            raise RuntimeError(error_msg) from e
        except Exception as e:
            raise RuntimeError(f"Error downloading video: {str(e)}") from e
    
    def list_videos(self, limit: Optional[int] = None) -> List[VideoTask]:
        """
        List all video generation tasks.
        
        Args:
            limit: Maximum number of tasks to return
            
        Returns:
            List of VideoTask objects
            
        Raises:
            requests.HTTPError: If API request fails
        """
        url = f"{self.base_url}/v1/videos"
        params = {}
        if limit is not None:
            params['limit'] = limit
        
        response = requests.get(url, headers=self._get_headers(), params=params, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        # Handle list response
        if isinstance(result, dict) and 'data' in result:
            tasks_data = result['data']
        elif isinstance(result, list):
            tasks_data = result
        else:
            raise ValueError(f"Unexpected API response format: {result}")
        
        return [VideoTask.from_dict(task_data) for task_data in tasks_data]
    
    def wait_for_completion(
        self,
        task_id: str,
        poll_interval: int = 5,
        timeout: int = 1800,
        callback: Optional[callable] = None
    ) -> VideoTask:
        """
        Wait for a video generation task to complete.
        
        Args:
            task_id: Task ID
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            callback: Optional callback function(task: VideoTask) called on each poll
            
        Returns:
            Completed VideoTask
            
        Raises:
            TimeoutError: If task doesn't complete within timeout
        """
        start_time = time.time()
        
        while True:
            task = self.get_task_status(task_id)
            
            if callback:
                callback(task)
            
            if task.status in ('completed', 'failed', 'error'):
                return task
            
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
            
            time.sleep(poll_interval)
