import logging
import os
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import v2
from torio.io import StreamingMediaDecoder
from tqdm import tqdm

from synchformer import Synchformer


logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

# Constants from eval_utils.py
_SYNC_SIZE = 224
_SYNC_FPS = 25.0

# Default device
device = 'cuda:0'

# Default paths
_syncformer_ckpt_path = Path(__file__).parent.parent / 'assets' / 'synchformer' / 'synchformer_state_dict.pth'


def error_avoidance_collate(batch):
    """Collate function that filters out None values."""
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)


class VideoDataset(Dataset):
    """Dataset for loading videos and extracting sync frames."""

    def __init__(
        self,
        video_paths: list[Path],
        *,
        duration_sec: float = 8.0,
    ):
        self.video_paths = video_paths
        self.duration_sec = duration_sec
        self.sync_expected_length = int(_SYNC_FPS * self.duration_sec)

        self.sync_transform = v2.Compose([
            v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
            v2.CenterCrop(_SYNC_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.video_paths)

    def sample(self, idx: int) -> dict[str, torch.Tensor]:
        video_path = self.video_paths[idx]

        reader = StreamingMediaDecoder(video_path)
        reader.add_basic_video_stream(
            frames_per_chunk=int(_SYNC_FPS * self.duration_sec),
            frame_rate=_SYNC_FPS,
            format='rgb24',
        )

        reader.fill_buffer()
        data_chunk = reader.pop_chunks()

        sync_chunk = data_chunk[0]
        
        if sync_chunk is None:
            raise RuntimeError(f'Sync video returned None {video_path}')
        
        if sync_chunk.shape[0] < self.sync_expected_length:
            # Pad with zero frames if shorter than expected
            
            missing_frames = self.sync_expected_length - sync_chunk.shape[0]
            zero_frames = torch.zeros(
                missing_frames,
                *sync_chunk.shape[1:],
                dtype=sync_chunk.dtype
            )
            sync_chunk = torch.cat([sync_chunk, zero_frames], dim=0)
            log.info(f"Pad {missing_frames} frames {video_path} to {sync_chunk.shape[0]}")

        # Truncate if longer than expected
        sync_chunk = sync_chunk[:self.sync_expected_length]
        if sync_chunk.shape[0] != self.sync_expected_length:
            raise RuntimeError(f'Sync video wrong length {video_path}, '
                               f'expected {self.sync_expected_length}, '
                               f'got {sync_chunk.shape[0]}')
        sync_chunk = self.sync_transform(sync_chunk)

        data = {
            'name': video_path.stem,
            'sync_video': sync_chunk,
        }
        return data

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        try:
            return self.sample(idx)
        except Exception as e:
            log.error(f'Error loading video {self.video_paths[idx]}: {e}')
            return None


def encode_video_with_sync(synchformer: Synchformer, x: torch.Tensor) -> torch.Tensor:
    """
    Encode video with synchformer.
    
    Args:
        synchformer: Synchformer model
        x: (B, T, C, H, W) tensor with H=W=224
    
    Returns:
        (B, S, T, D) tensor where S is number of segments, T is segment length, D is feature dim
    """
    # x: (B, T, C, H, W) H/W: 224
    b, t, c, h, w = x.shape
    assert c == 3 and h == 224 and w == 224

    # partition the video
    segment_size = 16
    step_size = 8
    num_segments = (t - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size:i * step_size + segment_size])
    x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

    ## avbenchmark
    # x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
    # x = synchformer.extract_vfeats(x)
    # x = rearrange(x, '(b s) 1 t d -> b s t d', b=b)

    ## mmaudio
    outputs = []
    batch_size = b * 40
    x = rearrange(x, 'b s t c h w -> (b s) 1 t c h w')
    for i in range(0, b * num_segments, batch_size):
        outputs.append(synchformer(x[i:i + batch_size]))
    x = torch.cat(outputs, dim=0)
    x = rearrange(x, '(b s) 1 t d -> b s t d', b=b)
    
    return x


@torch.inference_mode()
def extract(video_path, gt_cache, audio_length=8.0, num_workers=4, batch_size=8, device_id='cuda:0', synchformer_ckpt=None):
    """
    Extract synchformer features from videos.
    
    Args:
        video_path: Path to directory containing video files
        gt_cache: Path to save extracted features
        audio_length: Duration in seconds to extract (default: 8.0)
        num_workers: Number of data loading workers (default: 4)
        batch_size: Batch size for processing (default: 8)
        device_id: Device to use (default: 'cuda:0')
        synchformer_ckpt: Path to synchformer checkpoint (default: None, uses default path)
    """
    global device
    device = device_id
    
    # Convert to Path objects
    video_path = Path(video_path)
    gt_cache = Path(gt_cache)
    
    if synchformer_ckpt is None:
        synchformer_ckpt = _syncformer_ckpt_path
    else:
        synchformer_ckpt = Path(synchformer_ckpt)
    
    log.info('Extracting synchformer features...')
    
    # Read all video file names
    video_names = os.listdir(video_path)
    video_paths = [video_path / f for f in video_names if f.endswith('.mp4')]
    log.info(f'{len(video_paths)} videos found.')

    # Create dataset and dataloader
    dataset = VideoDataset(video_paths, duration_sec=audio_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=error_avoidance_collate,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Load synchformer model
    log.info(f'Loading synchformer from {synchformer_ckpt}')
    sync_model = Synchformer().to(device).eval()
    sd = torch.load(synchformer_ckpt, weights_only=True, map_location=device)
    sync_model.load_state_dict(sd)
    
    # Compile for better performance
    cmp_encode_video_with_sync = torch.compile(encode_video_with_sync)

    # Extract features and save as .npy files
    gt_cache.mkdir(parents=True, exist_ok=True)
    total_saved = 0
    
    for data in tqdm(loader, desc='Extracting features'):
        if data is None:
            continue
            
        name = data['name']
        sync_video = data['sync_video'].to(device)

        sync_features = cmp_encode_video_with_sync(sync_model, sync_video)
        sync_features = sync_features.cpu().detach()

        for i, n in enumerate(name):
            # Convert to numpy and save as .npy file for each video
            feature_numpy = sync_features[i].clone().cpu().numpy()
            
            # Save individual npy file for each video
            npy_path = gt_cache / f'{n}.npy'
            np.save(npy_path, feature_numpy)
            total_saved += 1

    log.info(f'Extracted and saved features for {total_saved} videos to {gt_cache}')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract synchformer features from videos')
    parser.add_argument('--video_path', type=str, required=True, help='Path to directory containing video files')
    parser.add_argument('--gt_cache', type=str, required=True, help='Path to save extracted features')
    parser.add_argument('--audio_length', type=float, default=8.0, help='Duration in seconds to extract (default: 8.0)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing (default: 8)')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (default: cuda:0)')
    parser.add_argument('--synchformer_ckpt', type=str, default=None, help='Path to synchformer checkpoint (default: uses default path)')
    
    args = parser.parse_args()
    
    extract(
        video_path=args.video_path,
        gt_cache=args.gt_cache,
        audio_length=args.audio_length,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        device_id=args.device,
        synchformer_ckpt=args.synchformer_ckpt
    )

