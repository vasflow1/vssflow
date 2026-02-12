import sys
import os
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = str(current_file.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from model.clip.clip_module import CLIPViT
import einops
import numpy as np
from PIL import Image
from glob import glob
import torch, torchaudio
import torchvision.io as tvio
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import decord
from tqdm import tqdm


def get_video_frames(
    va_path,
    video_process_config={'duration':10.0, 'resize_input_size': [224, 224], 'target_sampling_rate': 10, 'raw_duration_min_threshold':0.05},
    backend="decord"
):
    def check_and_drop(int_list, min_value, max_value):
        return [x for x in int_list if min_value <= x <= max_value]
    video_fram_h, video_frame_w = video_process_config['resize_input_size']
    clip_transform = transforms.Compose([
            transforms.Resize((video_fram_h, video_frame_w), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    video_process_config['target_frame_length'] = int(video_process_config['duration'] * video_process_config['target_sampling_rate'])

    if backend == "torchvision":
        try:
            frame_data, _, meta = tvio.read_video(va_path, pts_unit="sec", output_format="THWC")
            video_raw_fps = meta["video_fps"]
        except BaseException as e:
            raise e
        video_raw_frame_num = len(frame_data)
        video_raw_duration = video_raw_frame_num / video_raw_fps
        if video_raw_duration <= video_process_config['raw_duration_min_threshold']:
            raise RuntimeError(f"Video duration {video_raw_duration} is too short, less than {video_process_config['raw_duration_min_threshold']}.")
        video_sampled_frame_ids = [
            round(i * video_raw_fps / video_process_config['target_sampling_rate'])
            for i in range(video_process_config['target_frame_length'])
        ]
        video_sampled_frame_ids = check_and_drop(video_sampled_frame_ids, 0, video_raw_frame_num-1)
        sampled_frames = frame_data[video_sampled_frame_ids, :, :, :]
        if sampled_frames.dtype != torch.uint8:
            if sampled_frames.max() <= 1.0:
                sampled_frames = (sampled_frames * 255).byte()
            else:
                sampled_frames = sampled_frames.byte()
        sampled_frames = [Image.fromarray(frame.numpy()) for frame in sampled_frames]
    elif backend == "decord":
        try:
            decord_vr = decord.VideoReader(va_path, num_threads=1, ctx=decord.cpu(0))
        except BaseException as e:
            raise e
        video_raw_frame_num = len(decord_vr)
        video_raw_fps = decord_vr.get_avg_fps()
        video_raw_duration = video_raw_frame_num / video_raw_fps
        if video_raw_duration <= video_process_config['raw_duration_min_threshold']:
            raise RuntimeError(f"Video duration {video_raw_duration} is too short, less than {video_process_config['raw_duration_min_threshold']}.")
            
        # Sample frames with target sampling rate, 
        # NOTE Implemention of `FPS`` here is just selecting frames in raw 30fps video frames
        video_sampled_frame_ids = [ round(i * video_raw_fps / video_process_config['target_sampling_rate'])
                                    for i in range(video_process_config['target_frame_length']) ]
        video_sampled_frame_ids = check_and_drop(video_sampled_frame_ids, 0, video_raw_frame_num-1)
        sampled_frames = decord_vr.get_batch(video_sampled_frame_ids).asnumpy()
            
        if sampled_frames.dtype != np.uint8:
            if sampled_frames.max() <= 1.0:
                sampled_frames = (sampled_frames * 255).astype(np.uint8)
            else:
                sampled_frames = sampled_frames.astype(np.uint8)
            
        sampled_frames = [Image.fromarray(frame) for frame in sampled_frames]

    else:
        raise NotImplementedError(f"Backend `{backend}` is not implemented.")

    ''' Process frames '''
    p = video_process_config['target_frame_length'] - len(sampled_frames)
    if p > 0:
        w, h = sampled_frames[0].size  
        blank_pil = Image.new('RGB', (h, w), (0, 0, 0))
        sampled_frames = sampled_frames + [blank_pil] * p
    else:
        sampled_frames = sampled_frames[0 : video_process_config['target_frame_length']]
    assert len(sampled_frames) == video_process_config['target_frame_length'], f"Sampled frames length {len(sampled_frames)} is not equal to target length {video_process_config['target_frame_length']}."
    # Transform and stack
    sampled_frames = [clip_transform(sampled_frame) for sampled_frame in sampled_frames ]
    processed_frames = torch.stack(sampled_frames, dim=0)
    return processed_frames


def encode_image(image_encoder, image, use_projection=False):
    _b = image.shape[0]
    x = einops.rearrange(image, "b f c h w -> (b f) c h w")
    x = image_encoder(x)
    x = x[:, 0, :]
    return einops.rearrange(x, "(b f) c -> b f c", b=_b)


if __name__ == "__main__":
    batchsize = 4
    device = 'cuda:7'
    video_process_config={'duration':10.0, 'resize_input_size': [224, 224], 'target_sampling_rate': 10, 'raw_duration_min_threshold':0.05}
    
    # 设置输入视频路径和输出特征保存路径
    video_input_dir = '/home/chengxin/chengxin/dataset/V2C/dataset/**/*.mp4'
    output_dir = '/home/chengxin/chengxin/dataset/Dataset_feature/v2c_clip_b16'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载CLIP模型
    image_encoder = CLIPViT("/home/chengxin/chengxin/vasflow/assets/clip/ViT-B-16.pt")
    image_encoder = image_encoder.to(device)
    image_encoder.eval()
    
    # 获取所有视频路径
    video_paths = glob(video_input_dir, recursive=True)
    print(f"Total videos to process: {len(video_paths)}")
    
    # 批量处理视频
    for batch_idx in tqdm(range(0, len(video_paths), batchsize)):
        batch_paths = video_paths[batch_idx:batch_idx + batchsize]
        video_frames_list = []
        video_ids = []
        valid_indices = []
        
        # 读取当前批次的所有视频帧
        for idx, video_path in enumerate(batch_paths):
            try:
                frames = get_video_frames(video_path, video_process_config, backend="decord")
                video_frames_list.append(frames)
                video_id = ".".join((os.path.basename(video_path).split('.')[:-1]))
                video_ids.append(video_id)
                valid_indices.append(idx)
                print(f"Loaded video {batch_idx + idx + 1}/{len(video_paths)}: {video_id}")
            except Exception as e:
                print(f"Error loading video {video_path}: {e}")
                continue
        
        if len(video_frames_list) == 0:
            print(f"Batch {batch_idx // batchsize + 1}: No valid videos, skipping...")
            continue
        
        # 堆叠成批次并提取特征
        video_frames_batch = torch.stack(video_frames_list, dim=0).to(device)  # [B, F, C, H, W]
        with torch.no_grad():
            video_features = encode_image(image_encoder, video_frames_batch)  # [B, F, C]
        
        # 保存每个视频的特征
        for idx, video_id in enumerate(video_ids):
            feature = video_features[idx].cpu().numpy()  # [F, C]
            save_path = os.path.join(output_dir, f"{video_id}.npy")
            np.save(save_path, feature)
            print(f"Saved features for {video_id}: shape {feature.shape}")
        
        print(f"Batch {batch_idx // batchsize + 1} completed: processed {len(video_ids)} videos")
        
    
    print(f"All done! Features saved to {output_dir}")

    
