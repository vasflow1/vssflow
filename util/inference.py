import os, time, random, einops, wandb, inspect
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image


import torch, torchaudio
import torch.nn as nn
import torchvision.io as tvio
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


import yaml
import decord
from worker.vaflow_sda_dit_noise_text_mel import VAFlow
from worker.dp_tts import DurationPredictor
import torch.nn.functional as F
from util.get_dict import PhonemeTokenizer


def maxPathSumWithPath(matrix):
    m, n = len(matrix), len(matrix[0])
    dp = [[float('-inf')] * n for _ in range(m)]
    prev = [[None] * n for _ in range(m)]  # 记录前驱方向
    
    dp[0][0] = matrix[0][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + matrix[0][j]
        prev[0][j] = 'left'
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + matrix[i][0]
        prev[i][0] = 'up'
    for i in range(1, m):
        for j in range(1, n):
            max_val = float('-inf')
            direction = None
            
            if i > 0 and dp[i-1][j] > max_val:
                max_val = dp[i-1][j]
                direction = 'up'
            if j > 0 and dp[i][j-1] > max_val:
                max_val = dp[i][j-1]
                direction = 'left'
            if i > 0 and j > 0 and dp[i-1][j-1] > max_val:
                max_val = dp[i-1][j-1]
                direction = 'diag'
            
            dp[i][j] = max_val + matrix[i][j]
            prev[i][j] = direction
    

    path = []
    i, j = m-1, n-1
    while i >= 0 and j >= 0:
        path.append((i, j))
        if prev[i][j] == 'up':
            i -= 1
        elif prev[i][j] == 'left':
            j -= 1
        elif prev[i][j] == 'diag':
            i -= 1
            j -= 1
        else:
            break 
    
    path.reverse()  
    return dp[m-1][n-1], path



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
    assert len(sampled_frames) == video_process_config['target_frame_length'], f"Sampled frames length {len(sampled_frames)} is not equal to target length {self.video_target_frame_len}."
    # Transform and stack
    sampled_frames = [clip_transform(sampled_frame) for sampled_frame in sampled_frames ]
    processed_frames = torch.stack(sampled_frames, dim=0)
    return processed_frames




# def phonemes_to_token_ids(phoneme_seq, config_path='/home/chengxin/chengxin/Dataset_Sound/MetaData/vaflow2_meta/meta/token_list.json'):
#     with open(config_path, 'r') as f:
#         phoneme2id = yaml.safe_load(f)
#     token_ids = [phoneme2id.get(p, phoneme2id.get('<blank>', 0)) for p in phoneme_seq]
#     return torch.tensor(token_ids, dtype=torch.long)


def text_to_token_ids(text_seq, config_path='/home/chengxin/chengxin/vasflow/util/token_list.json'):
    tokenizer = PhonemeTokenizer(g2p_type="g2p_en_no_space", non_linguistic_symbols=None,)
    with open(config_path, 'r') as f:
        phoneme2id = yaml.safe_load(f)
    phoneme_seq = []
    token_ids = []

    for text in text_seq:
        phonemes = ["<blank>"] + tokenizer.text2tokens(text) + ["<blank>"]
        token_id = [phoneme2id.get(p, phoneme2id.get('<blank>', 0)) for p in phonemes]

        phoneme_seq.append(phonemes)
        token_ids.append(torch.tensor(token_id, dtype=torch.long))
    return token_ids, phoneme_seq


def pad_sequence(sequences, padding_value=0, max_length=None, pad_type='right', target_dim=None):
    if not sequences:
        raise ValueError("sequences should not be empty")

    # Determine which dimension to pad (default: 0)
    if target_dim is None:
        target_dim = 0

    # Find max length
    lengths = [s.shape[target_dim] for s in sequences]
    pad_len = max(lengths) if max_length is None else max_length


    # Pad each tensor
    padded = []
    for s in sequences:
        pad_size = pad_len - s.shape[target_dim]
        if pad_size < 0:
            raise ValueError("Some sequence is longer than pad_len")
        pad_shape = [0] * (2 * s.dim())
        pad_shape[2 * (s.dim() - target_dim) - 1] = pad_size  # pad right
        if pad_type == 'left':
            pad_shape[2 * (s.dim() - target_dim) - 2] = pad_size  # pad left
            pad_shape[2 * (s.dim() - target_dim) - 1] = 0
        s_padded = torch.nn.functional.pad(s, pad_shape, mode='constant', value=padding_value)
        padded.append(s_padded)
    return torch.stack(padded, dim=0)



def adjust_speech(speech_start_duration, phone_id, duration_matrix, device):
    # phone_id [phone_length]    duration_matrix [250, phone_length]
    # phone_id = torch.cat((torch.tensor([59]).to(phone_id), phone_id))
    if speech_start_duration == 0:
        return torch.cat((torch.tensor([0], dtype=torch.int).to(phone_id), phone_id)), \
                torch.cat([duration_matrix, torch.zeros([250 ,1]).to(duration_matrix)], dim = -1)
    
    
    assert phone_id[0] == 59
    phone_id = torch.cat((torch.tensor([0], dtype=torch.int).to(phone_id), phone_id))            # [1 + phone_length]
    silence_length = int(round(speech_start_duration*25, 0))
    duration_matrix_new = torch.zeros([silence_length, phone_id.shape[-1]], device=device)     # [append_dur_length, 1 + phone_length]
    duration_matrix_new[:, 0] = 1

    assert duration_matrix[-silence_length].sum() == 0
    duration_matrix = duration_matrix[:-silence_length]                              # [250 - append_dur_length, phone_length]
    duration_matrix = F.pad(duration_matrix, (1, 0), mode='constant', value=0)       # [250 - append_dur_length, 1 + phone_length]
    duration_matrix_new = torch.cat((duration_matrix_new, duration_matrix), dim=0)   # [250, 1 + phone_length]
    return phone_id, duration_matrix_new




def infer_videos(vaflow_model, 
                 dp_model,
                save_path: str,
                mp4_paths:list[str], 
                video_process_config = {'duration':10.0, 'resize_input_size': [224, 224], 'target_sampling_rate': 10, 'raw_duration_min_threshold':0.05},
                text_seq = None,
                avhubert_feature_paths = None,
                speech_durations = None,
                speech_start_durations = None,
                ref_audio_ebd_paths = None,
                guidance_scale = 3,
                num_samples_per_prompt = 1,
                device="cuda"):
    vaflow_model.to(device)
    dp_model.to(device)
    batchsize = len(mp4_paths)


    # SPEECH PHONEMES
    phone_id = torch.zeros([len(mp4_paths), 1], dtype=torch.int, device=device)
    duration_matrix = torch.zeros(len(mp4_paths), vaflow_model.latent_length, phone_id.shape[-1], device=device)
    ref_audio_ebd = torch.zeros(len(mp4_paths), 256, device=device)
    phoneme_seq = None

    if text_seq is not None:
        if ref_audio_ebd_paths is not None:
            ref_audio_ebd = torch.stack([torch.tensor(np.load(path)).to(device) for path in ref_audio_ebd_paths])
        token_ids, phoneme_seq = text_to_token_ids(text_seq)
        phone_id = pad_sequence(token_ids, padding_value=0).to(device)  # [B, max_seq_len]
        phone_length = torch.tensor([len(seq) for seq in token_ids], dtype=torch.long, device=device)  # [B]
        
        # Prepare avhubert features and pad
        avhubert_list = []
        avhubert_length = []
        if avhubert_feature_paths is not None:
            for path in avhubert_feature_paths:
                feat = torch.tensor(np.load(path), device=device)
                avhubert_list.append(feat)
                avhubert_length.append(feat.shape[0])
        else:
            assert speech_durations is not None
            for dur in speech_durations:
                feat = torch.ones(int(25 * dur), 1024, device=device)
                avhubert_list.append(feat)
                avhubert_length.append(feat.shape[0])

        avhubert = pad_sequence(avhubert_list, padding_value=0.0, max_length = 250)  # [B, max_avhubert_len, 1024]
        avhubert_length = torch.tensor(avhubert_length, dtype=torch.long, device=device)  # [B]
        batch = {
            "video_id": None,
            "duration_matrix": None,
            "duration_span": None,
            "avhubert": avhubert,
            "avhubert_length": avhubert_length,
            "phone_id": phone_id,
            "phone_length": phone_length,
        }
        attns = dp_model.predict_step(batch)
        print(attns.shape)

        for i in range(attns.shape[0]):
            max_sum, path = maxPathSumWithPath(attns[i]-1)
            attn = torch.zeros_like(attns[i])
            for p in path:
                attn[p[0], p[1]] = 1
            attn[int(avhubert_length[i]):] = 0
            attns[i] = attn
        
        duration_matrix = attns
        if speech_start_durations is not None:
            new_phone_ids = torch.cat([phone_id, torch.zeros([batchsize,1]).to(phone_id)], dim = -1)
            new_duration_matrixs = torch.cat([duration_matrix, torch.zeros([batchsize, 250 ,1]).to(duration_matrix)], dim = -1)
            for i in range(duration_matrix.shape[0]):
                new_phone_id, new_duration_matrix = adjust_speech(speech_start_durations[i], phone_id[i], duration_matrix[i], device)
                print(phone_id.shape, new_phone_id.shape, duration_matrix.shape, new_duration_matrix.shape)
                new_phone_ids[i] = new_phone_id
                new_duration_matrixs[i] = new_duration_matrix
            phone_id = new_phone_ids
            duration_matrix = new_duration_matrixs


    # VIDEO FRAMES
    video_frames_list = []
    video_ids = []
    for path in mp4_paths:
        frames = get_video_frames(path, video_process_config)
        video_frames_list.append(frames)
        video_ids.append(".".join((os.path.basename(path).split('.')[:-1])))
    video_frames_batch = torch.stack(video_frames_list, dim=0).to(device)  # [B, F, C, H, W]
    with torch.no_grad():
        video_features = vaflow_model.encode_image(video_frames_batch)  # [B, F, C]
    # video_features = torch.zeros_like(video_features).to(video_features)


    # INFERENCE
    batch = {
        "video_feat": video_features,
        "video_id": video_ids,
        "video_path": mp4_paths,
        "ref_audio_ebd": ref_audio_ebd,                                      # NOT IMPLEMENETED
        "audio_waveform": torch.zeros(len(mp4_paths), 1, vaflow_model.audio_sample_rate * 10, device=device),  # Dont Need
        "audio_sr": [16000] * len(mp4_paths),
        "audio_duration": [10.0] * len(mp4_paths),
        "duration_matrix": duration_matrix,  
        "phone_id": phone_id,                   
        "phone_seq": phoneme_seq,                                                      
    }
    os.makedirs(save_path, exist_ok=True)
    vaflow_model.val_log_dir_for_video_per_epoch = save_path
    vaflow_model.num_samples_per_prompt = num_samples_per_prompt
    vaflow_model.guidance_scale = guidance_scale
    vaflow_model.validation_step(batch, batch_idx=0, dataloader_idx=0)
    # print(torch.bmm(duration_matrix , phone_id.unsqueeze(-1).to(torch.float32)))





