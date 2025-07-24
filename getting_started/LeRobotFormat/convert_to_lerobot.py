
#!/usr/bin/env python3
"""
Convert Isaac Sim Franka recordings to LeRobot format.

This script converts data from the following structure:
    /mnt/data/franka_recordings/episode_XXX/
        episode_data.npz
        left_camera.mp4
        right_camera.mp4
        wrist_camera.mp4
        ...

To the LeRobot format:
    <output_dir>/
    ├── data/
    │   └── chunk-000/
    │       ├── episode_000000.parquet
    │       └── ...
    ├── videos/
    │   └── chunk-000/
    │       ├── observation.images.left/
    │       │   ├── episode_000000.mp4
    │       │   └── ...
    │       ├── observation.images.right/
    │       ├── observation.images.wrist/
    ├── meta/
    │   ├── info.json
    │   └── episodes.jsonl
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import cv2
from typing import Any

# ----------------- Image sampling & statistics helpers -----------------

def sample_indices(n: int, num_samples: int = 100) -> np.ndarray:
    """Return sorted random indices for sampling."""
    if n <= num_samples:
        return np.arange(n)
    return np.sort(np.random.choice(n, size=num_samples, replace=False))


def sample_frames(video_path: Path, num_samples: int = 100) -> np.ndarray | None:
    """Sample frames from a video file and return as uint8 array (N, C, H, W)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0 or not cap.isOpened():
        cap.release()
        return None
    indices = sample_indices(total, num_samples)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # HWC, RGB, uint8
        frame = np.transpose(frame, (2, 0, 1))  # CHW
        frames.append(frame)
    cap.release()
    if len(frames) == 0:
        return None
    return np.asarray(frames, dtype=np.uint8)


def get_feature_stats(array: np.ndarray, axis: tuple, keepdims: bool) -> dict[str, np.ndarray]:
    return {
        "min": np.min(array, axis=axis, keepdims=keepdims),
        "max": np.max(array, axis=axis, keepdims=keepdims),
        "mean": np.mean(array, axis=axis, keepdims=keepdims),
        "std": np.std(array, axis=axis, keepdims=keepdims),
        "count": np.array([array.shape[0]]),
    }


def aggregate_feature_stats(stats_ft_list: list[dict]) -> dict[str, np.ndarray]:
    """Aggregate stats across episodes for a single feature (image/video)."""
    # Convert all inputs to numpy arrays if they aren't already
    means = np.stack([np.asarray(s["mean"]) for s in stats_ft_list])
    stds = np.stack([np.asarray(s["std"]) for s in stats_ft_list])
    counts = np.stack([np.asarray(s["count"]) for s in stats_ft_list])
    
    # Calculate weighted mean and variance
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    total_mean = (means * counts).sum(axis=0) / total_count
    
    # Calculate combined variance using the parallel algorithm
    delta_means = means - total_mean
    total_variance = ((stds**2 + delta_means**2) * counts).sum(axis=0) / total_count
    
    return {
        "min": np.min(np.stack([np.asarray(s["min"]) for s in stats_ft_list]), axis=0),
        "max": np.max(np.stack([np.asarray(s["max"]) for s in stats_ft_list]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }
# -----------------------------------------------------------------------

# Configuration
INPUT_DIR = "/mnt/data/franka_recordings" #"/mnt/data/test_data" 
OUTPUT_DIR = "/mnt/data/franka_lerobot_dataset"

# Camera mapping from source to LeRobot format
CAMERA_MAPPING = {
    'left': 'left',
    'right': 'right',
    'wrist': 'wrist'
}

@dataclass
class EpisodeMetadata:
    """Metadata for a single episode."""
    episode_id: int
    duration: float
    num_frames: int
    start_time: float
    end_time: float
    camera_views: List[str]

class LeRobotConverter:
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize the converter with input and output directories."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.episode_dirs = sorted([d for d in self.input_dir.glob('episode_*') if d.is_dir()])
        self.metadata = []
        self.image_stats_per_episode: list[dict[str, dict[str, np.ndarray]]] = []
        self.global_index = 0  # Track global index across all episodes
        
        # Create output directories
        (self.output_dir / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'videos' / 'chunk-000').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'meta').mkdir(parents=True, exist_ok=True)
    
    def process_episode(self, episode_dir: Path, episode_idx: int) -> Optional[Dict]:
        """Process a single episode directory."""
        print(f"Processing {episode_dir.name}...")
        
        # Check for required files
        npz_file = episode_dir / 'episode_data.npz'
        if not npz_file.exists():
            print(f"Skipping {episode_dir.name}: episode_data.npz not found")
            return None
        
        # Load episode data
        try:
            data = np.load(npz_file, allow_pickle=True)
        except Exception as e:
            print(f"Error loading {npz_file}: {e}")
            return None
        
        # Process video files and compute image statistics
        camera_views = []
        episode_image_stats: dict[str, dict[str, np.ndarray]] = {}
        for cam_name in CAMERA_MAPPING.keys():
            video_file = episode_dir / f"{cam_name}_camera.mp4"
            if video_file.exists():
                camera_views.append(cam_name)
                stats = self._process_video(video_file, cam_name, episode_idx)
                if stats is not None:
                    episode_image_stats[f"observation.images.{cam_name}"] = stats
        
        # Create episode metadata
        num_frames = len(data['timestamps']) if 'timestamps' in data else 0
        duration = (data['timestamps'][-1] - data['timestamps'][0]) if num_frames > 0 else 0
        
        episode_meta = EpisodeMetadata(
            episode_id=episode_idx,
            duration=float(duration),
            num_frames=num_frames,
            start_time=0,
            end_time=float(data['timestamps'][-1] - data['timestamps'][0]) if num_frames > 0 else 0.0,
            camera_views=camera_views
        )
        
        # Convert data to LeRobot format
        self._convert_to_parquet(data, episode_idx)
        # Save image stats for this episode
        self.image_stats_per_episode.append(episode_image_stats)
        
        return asdict(episode_meta)
    
    def _process_video(self, video_file: Path, camera_name: str, episode_idx: int):
        """Process a single video file."""
        target_dir = self.output_dir / 'videos' / 'chunk-000' / f'observation.images.{camera_name}'
        target_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = target_dir / f'episode_{episode_idx:06d}.mp4'
        
        # For now, just copy the file. In a real implementation, you might want to:
        # 1. Check video properties
        # 2. Transcode if necessary
        # 3. Add metadata
        shutil.copy2(video_file, target_file)
        print(f"  Copied {video_file.name} to {target_file}")
        # -------- sample frames & compute stats --------
        frames = sample_frames(video_file)
        if frames is None:
            print(f"  Warning: unable to sample frames from {video_file}")
            return None
        # Compute stats over sampled frames (normalize to 0-1, keep channel dim)
        stats = get_feature_stats(frames.astype(np.float32) / 255.0, axis=(0, 2, 3), keepdims=True)
        return stats
    
    def _convert_to_parquet(self, data: Dict, episode_idx: int):
        """Convert episode data to parquet format."""
        # Initialize lists to store data
        observations = []
        actions = []
        timestamps = []
        
        # Convert timestamps to relative (starting from 0)
        start_time = float(0.0)

        
        # Process each timestep
        for i in range(len(data['timestamps'])):
            # Create observation dictionary
            obs = data['joint_states'][i].tolist() + data['gripper_states'][i].tolist()
            
            # Create action dictionary (in a real implementation, you might compute this)
            action = data['gello_joint_states'][i].tolist() + data['gripper_states'][i].tolist()              
            observations.append(obs)
            actions.append(action)
            # Store relative timestamp (seconds since start of episode)
            timestamps.append(float(data['timestamps'][i] - data['timestamps'][0]))
        
        # Calculate indices
        num_frames = len(timestamps)
        frame_indices = np.arange(num_frames)
        global_indices = np.arange(self.global_index, self.global_index + num_frames)
        
        # Update global index for next episode
        self.global_index += num_frames
        
        # Create a DataFrame with the episode data
        df = pd.DataFrame({
            'observation.state': observations,
            'action': actions,
            'timestamp': timestamps,
            'frame_index': frame_indices,
            'episode_index': np.full(num_frames, episode_idx, dtype=np.int32),
            'index': global_indices,
            'task_index': np.zeros(num_frames, dtype=np.int32),
            'annotation.human.action.task_description':  np.zeros(num_frames, dtype=np.int32)
        })
        
        # Convert to pyarrow table and write to parquet
        table = pa.Table.from_pandas(df, preserve_index=False)
        output_file = self.output_dir / 'data' / 'chunk-000' / f'episode_{episode_idx:06d}.parquet'
        pq.write_table(table, output_file)
        print(f"  Saved parquet data to {output_file}")
    
    def _generate_info_json(self):
        """Generate the info.json file with dataset metadata."""
        total_frames = sum(ep['num_frames'] for ep in self.metadata)
        total_duration = sum(ep['duration'] for ep in self.metadata)
        
        info = {
            "codebase_version": "v2.1",
            "robot_type": "franka_emika_panda",
            "fps": 30,  # Assuming 30 FPS, adjust if different
            "total_episodes": len(self.metadata),
            "total_frames": total_frames,
            "total_tasks": 1,  # Assuming single task for now
            "total_chunks": 1,
            "chunks_size": 1000,
            "total_videos": len(self.metadata) * len(CAMERA_MAPPING),
            "splits": {
                "train": f"0:{len(self.metadata)}"  # All episodes for training
            },
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.state": {
                    "dtype": "float32",
                    "shape": [8],  #7+1 Adjust based on your state space
                    "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]
                },
                "action": {
                    "dtype": "float32",
                    "shape": [8],  # Adjust based on your action space
                    "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7", "gripper"]
                },
                "timestamp": {
                    "dtype": "float32",
                    "shape": [1]
                },
                "frame_index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "task_index": {
                    "dtype": "int64",
                    "shape": [1]
                },
                "annotation.human.action.task_description": {
                    "dtype": "int64",
                    "shape": [1]
                }
            },
            "camera_keys": [f"observation.images.{cam}" for cam in CAMERA_MAPPING.values()]
        }
        
        # Add video features
        for cam in CAMERA_MAPPING.values():
            info["features"][f"observation.images.{cam}"] = {
                "dtype": "video",
                "shape": [480, 640, 3],  # Will be updated with actual dimensions
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": 30,
                    "video.codec": "mp4v"
                }
            }
        
        with open(self.output_dir / 'meta' / 'info.json', 'w') as f:
            json.dump(info, f, indent=2)
    
    def _generate_episodes_jsonl(self):
        """Generate the episodes.jsonl file with per-episode metadata."""
        with open(self.output_dir / 'meta' / 'episodes.jsonl', 'w') as f:
            for ep in self.metadata:
                episode_info = {
                    "episode_index": ep['episode_id'],
                    "tasks": ["Pick the bowl and place it in the green square"],  # Default task
                    "length": ep['num_frames'],
                    # "camera_views": ep['camera_views']
                }
                f.write(json.dumps(episode_info) + '\n')
    
    def _generate_tasks_jsonl(self):
        """Generate the tasks.jsonl file with task information."""
        with open(self.output_dir / 'meta' / 'tasks.jsonl', 'w') as f:
            task_info = {
                "task_index": 0,
                "task": "pick_and_place",
                "description": "Pick and place objects with the Franka robot"
            }
            f.write(json.dumps(task_info) + '\n')
    
    def _compute_episode_stats(self, episode_idx):
        """Compute statistics for a single episode."""
        # Load the parquet file to compute stats
        parquet_path = self.output_dir / 'data' / 'chunk-000' / f'episode_{episode_idx:06d}.parquet'
        df = pd.read_parquet(parquet_path)
        
        stats = {
            "episode_index": episode_idx,
            "stats": {}
        }
        
        frame_count = len(df)
        # Compute stats for observation.state
        if 'observation.state' in df.columns:
            states = np.stack(df['observation.state'].values)
            stats["stats"]["observation.state"] = {
                "min": states.min(axis=0).tolist(),
                "max": states.max(axis=0).tolist(),
                "mean": states.mean(axis=0).tolist(),
                "std": states.std(axis=0).tolist(),
                "count": [frame_count]
            }
        
        # Compute stats for action
        if 'action' in df.columns:
            actions = np.stack(df['action'].values)
            stats["stats"]["action"] = {
                "min": actions.min(axis=0).tolist(),
                "max": actions.max(axis=0).tolist(),
                "mean": actions.mean(axis=0).tolist(),
                "std": actions.std(axis=0).tolist(),
                "count": [frame_count]
            }
        
                # Add image stats
        if episode_idx < len(self.image_stats_per_episode):
            for key, val in self.image_stats_per_episode[episode_idx].items():
                # overwrite count with actual frame count
                val_copy = {k: (v if isinstance(v, list) else v.tolist()) for k, v in val.items()}
                val_copy["count"] = [frame_count]
                stats["stats"][key] = val_copy
        return stats
    
    def _generate_episodes_stats_jsonl(self):
        """Generate the episodes_stats.jsonl file with per-episode statistics."""
        with open(self.output_dir / 'meta' / 'episodes_stats.jsonl', 'w') as f:
            for ep in self.metadata:
                stats = self._compute_episode_stats(ep['episode_id'])
                f.write(json.dumps(stats) + '\n')
    
    def _generate_stats_json(self):
        """Generate the stats.json file with dataset-wide statistics."""
        all_stats = []
        for ep in self.metadata:
            stats = self._compute_episode_stats(ep['episode_id'])
            all_stats.append(stats["stats"])
        
        # Aggregate stats across all episodes (numeric and image)
        aggregated_stats = {}
        data_keys = {key for s in all_stats for key in s}
        for key in data_keys:
            if key.startswith("observation.images."):
                agg = aggregate_feature_stats([s[key] for s in all_stats if key in s])
                aggregated_stats[key] = {k: v.tolist() for k, v in agg.items()}
                continue
            min_vals = np.min([s[key]["min"] for s in all_stats], axis=0)
            max_vals = np.max([s[key]["max"] for s in all_stats], axis=0)
            mean_vals = np.mean([s[key]["mean"] for s in all_stats], axis=0)
            std_vals = np.mean([s[key]["std"] for s in all_stats], axis=0)
            count_vals = np.sum([s[key]["count"] for s in all_stats], axis=0)
            
            aggregated_stats[key] = {
                "min": min_vals.tolist(),
                "max": max_vals.tolist(),
                "mean": mean_vals.tolist(),
                "std": std_vals.tolist(),
                "count": count_vals.tolist()
            }
        
        with open(self.output_dir / 'meta' / 'stats.json', 'w') as f:
            json.dump(aggregated_stats, f, indent=2)
    
    def generate_metadata(self):
        """Generate all metadata files for the dataset."""
        print("Generating metadata files...")
        
        # Generate all metadata files
        self._generate_info_json()
        self._generate_episodes_jsonl()
        self._generate_tasks_jsonl()
        self._generate_episodes_stats_jsonl()
        self._generate_stats_json()
        
        # Create a README.md
        with open(self.output_dir / 'meta' / 'README.md', 'w') as f:
            f.write("# LeRobot Dataset\n\n")
            f.write("This dataset was converted from Isaac Sim Franka recordings.\n\n")
            f.write("## Dataset Structure\n")
            f.write("- `data/`: Contains the parquet files with episode data\n")
            f.write("- `videos/`: Contains the video files for each camera view\n")
            f.write("- `meta/`: Contains metadata and statistics\n\n")
            f.write("## Metadata Files\n")
            f.write("- `info.json`: Dataset metadata and feature descriptions\n")
            f.write("- `episodes.jsonl`: Per-episode metadata\n")
            f.write("- `tasks.jsonl`: Task descriptions\n")
            f.write("- `episodes_stats.jsonl`: Per-episode statistics\n")
            f.write("- `stats.json`: Dataset-wide statistics\n")
        
        print(f"Generated metadata in {self.output_dir / 'meta'}")
    
    def run(self):
        """Run the conversion process."""
        print(f"Starting conversion of {len(self.episode_dirs)} episodes...")
        
        # Process each episode
        for i, episode_dir in enumerate(self.episode_dirs):
            episode_meta = self.process_episode(episode_dir, i)
            if episode_meta:
                self.metadata.append(episode_meta)
        
        # Generate metadata files
        self.generate_metadata()
        
        print(f"\nConversion complete! Output saved to {self.output_dir}")

if __name__ == "__main__":
    converter = LeRobotConverter(INPUT_DIR, OUTPUT_DIR)
    converter.run()
