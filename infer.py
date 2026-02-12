# VAFlow Inference Script
# 输入: video, transcript, reference audio
# 输出: 生成的 audio 和 speech

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import re
from pathlib import Path
import numpy as np
import torch
import torchaudio
import torch.nn.functional as nn_func
import yaml
from tqdm import tqdm
import copy
import math


sys.path.insert(0, '/home/chengxin/chengxin/vssflow')
sys.path.insert(0, '/home/chengxin/chengxin/vssflow/feature')

# 导入必要的模块
from worker.vaflow_noise_lip_synch_text import VAFlow, WrappedModel
from flow_matching.solver import ODESolver
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor
from feature.extract_phoneme import TextCleaner, PhonemeTokenizer
from feature.extract_rawnet import extract_speaker_embd
from feature.extract_clip import get_video_frames, encode_image
from feature.extract_synchformer import VideoDataset, encode_video_with_sync
from model.clip.clip_module import CLIPViT
from RawNet.python.RawNet3.models.RawNet3 import RawNet3
from RawNet.python.RawNet3.models.RawNetBasicBlock import Bottle2neck
import soundfile as sf
from einops import rearrange
from synchformer import Synchformer

print("Imports completed!")






# ========== 配置参数 ==========
# 模型配置
CONFIG = {
    # 模型路径
    "ckpt_dir_vae":"./assets/vae",
    "ckpt_dir_image_encoder": "./assets/clip/ViT-B-16.pt",
    "ckpt_dir_audio_dit": "./assets/stable_audio/ckpt/transformer_lip_synch_text",
    "ckpt_dir_vocoder": "./assets/vocoder",
    "vaflow_ckpt_path": "./log/2025_12_28-22_22_24-vaflow_noise_lip_synch_text_v2c_synthesized/ckpt/epoch=0199-step=6.00e+03.ckpt",  # 修改为你的checkpoint路径
    "rawnet_ckpt_path": "./feature/RawNet/python/RawNet3/models/weights/model.pt",
    "synchformer_ckpt_path": "./assets/synchformer/synchformer_state_dict.pth",
    "token_list_path": "./data/token_list.json",  # phoneme到id的映射文件
    
    # 模型参数
    "phone_ebd_dim": 32,
    "cond_feat_dim": 768,
    "lip_feat_dim": 1024,
    "synch_feat_dim": 768,
    "dit_num_layers": 10,
    "original_channel": 128,
    "audio_duration_sec": 10,
    "audio_length_per_sec": 25,
    "vae_latent_scaling_factor": 1.0,
    "scale_factor": 1.0,
    
    # 推理参数
    "guidance_scale": 3.0,
    "sample_steps": 10,
    "sample_method": "dopri5",
    "audio_sample_rate": 16000,
    "num_samples_per_prompt": 1,
    
    # 设备
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# 视频处理配置
VIDEO_PROCESS_CONFIG = {
    'duration': 10.0,
    'resize_input_size': [224, 224],
    'target_sampling_rate': 10,
    'raw_duration_min_threshold': 0.05
}

print(f"Configuration loaded. Device: {CONFIG['device']}")






# ========== 加载模型 ==========
print("Loading models...")

# 加载VAFlow模型
vaflow_model = VAFlow(
    ckpt_dir_image_encoder=CONFIG["ckpt_dir_image_encoder"],
    ckpt_dir_audio_dit=CONFIG["ckpt_dir_audio_dit"],
    ckpt_dir_vocoder=CONFIG["ckpt_dir_vocoder"],
    vaflow_ckpt_path=CONFIG["vaflow_ckpt_path"],
    phone_ebd_dim=CONFIG["phone_ebd_dim"],
    cond_feat_dim=CONFIG["cond_feat_dim"],
    lip_feat_dim=CONFIG["lip_feat_dim"],
    synch_feat_dim=CONFIG["synch_feat_dim"],
    dit_num_layers=CONFIG["dit_num_layers"],
    ckpt_dir_vae=CONFIG['ckpt_dir_vae'],
    vae_latent_scaling_factor=CONFIG["vae_latent_scaling_factor"],
    original_channel=CONFIG["original_channel"],
    audio_duration_sec=CONFIG["audio_duration_sec"],
    audio_length_per_sec=CONFIG["audio_length_per_sec"],
    scale_factor=CONFIG["scale_factor"],
    guidance_scale=CONFIG["guidance_scale"],
    sample_steps=CONFIG["sample_steps"],
    sample_method=CONFIG["sample_method"],
    audio_sample_rate=CONFIG["audio_sample_rate"],
    num_samples_per_prompt=CONFIG["num_samples_per_prompt"],
    resume_training=False,
    ignore_keys=[],
)

vaflow_model = vaflow_model.to(CONFIG["device"])
vaflow_model.eval()
print("VAFlow model loaded!")

# 加载RawNet模型（用于提取speaker embedding）
rawnet_model = RawNet3(
    Bottle2neck,
    model_scale=8,
    context=True,
    summed=True,
    encoder_type="ECA",
    nOut=256,
    out_bn=False,
    sinc_stride=10,
    log_sinc=True,
    norm_sinc="mean",
    grad_mult=1,
)
rawnet_model.load_state_dict(torch.load(CONFIG["rawnet_ckpt_path"], map_location="cpu")["model"])
rawnet_model = rawnet_model.to(CONFIG["device"])
rawnet_model.eval()
print("RawNet model loaded!")

clip_model = CLIPViT(CONFIG["ckpt_dir_image_encoder"])
clip_model = clip_model.to(CONFIG["device"])
clip_model.eval()
print("CLIP model loaded!")

synchformer_model = Synchformer().to(CONFIG["device"]).eval()
sd = torch.load(CONFIG["synchformer_ckpt_path"], weights_only=True, map_location=CONFIG["device"])
synchformer_model.load_state_dict(sd)

print("Synchformer model loaded!")
print("All models loaded successfully!")






# ========== 特征提取函数 ==========

def text_to_phoneme_ids(text, token_list_path, 
                        speech_start_sec = 0, 
                        speech_end_sec = 10,
                        audio_duration_sec=10, 
                        audio_length_per_sec=25):
    """将文本转换为phoneme id序列"""
    # 加载phoneme到id的映射
    with open(token_list_path, 'r') as f:
        phoneme2id = yaml.safe_load(f)
    
    # 文本清理和phoneme转换
    def add_spaces_around_digits(input_string):
        return re.sub(r'(?<=\d)(?=[a-zA-Z])|(?<=[a-zA-Z])(?=\d)', r' \g<0> ', input_string)
    
    cleaner = TextCleaner("tacotron")
    tokenizer = PhonemeTokenizer(g2p_type="g2p_en_no_space", non_linguistic_symbols=None)
    
    # 处理文本
    text = add_spaces_around_digits(text)
    text = cleaner(text)
    phonemes = ["<blank>"] + tokenizer.text2tokens(text) + ["<blank>"]
    
    # 转换为id
    phone_ids = [phoneme2id.get(p, phoneme2id.get('<blank>', 0)) for p in phonemes]
    phone_ids = torch.tensor(phone_ids, dtype=torch.long)

    # Speech的长度
    target_len = int((speech_end_sec - speech_start_sec) * audio_length_per_sec)
    repeat_factor = math.ceil(target_len / phone_ids.shape[0])
    phone_ids = phone_ids.repeat_interleave(repeat_factor, dim=0)   
    phone_ids = nn_func.interpolate(phone_ids[None, None, :].float(), size=target_len, mode="nearest").squeeze().long()
    phone_ids = torch.concat([torch.zeros([int(speech_start_sec * audio_length_per_sec)]), phone_ids])
    
    
    # 调整长度到latent_length
    latent_length = int(audio_duration_sec * audio_length_per_sec)
    if len(phone_ids) > latent_length:
        phone_ids = phone_ids[:latent_length]
    else:
        phone_ids = torch.nn.functional.pad(phone_ids, (0, latent_length - len(phone_ids)), mode='constant', value=0)
    
    return phone_ids, phonemes


def extract_ref_audio_embedding(audio_path, rawnet_model, device, n_samples=48000, n_segments=10):
    """从reference audio提取speaker embedding"""
    original_audio_path = audio_path
    temp_file_created = False
    
    # 读取音频
    audio, sample_rate = sf.read(audio_path)
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)  # 转为单声道
    
    # 重采样到16kHz（如果需要）
    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
        # 保存临时16kHz文件
        import tempfile
        tmp_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        sf.write(tmp_path.name, audio, 16000)
        audio_path = tmp_path.name
        tmp_path.close()
        temp_file_created = True
    
    # 提取embedding（extract_speaker_embd需要文件路径）
    embedding = extract_speaker_embd(
        rawnet_model,
        audio_path,
        n_samples=n_samples,
        n_segments=n_segments,
        gpu=(device != "cpu")
    )
    
    # 清理临时文件
    if temp_file_created:
        try:
            os.unlink(audio_path)
        except:
            pass
    
    # 返回平均embedding
    if isinstance(embedding, torch.Tensor):
        return embedding.mean(0).cpu().numpy()
    else:
        return embedding.mean(0)


def extract_video_clip_features(video_path, clip_model, device, video_process_config):
    """从video提取CLIP特征"""
    # 获取视频帧
    frames = get_video_frames(video_path, video_process_config, backend="decord")
    frames = frames.unsqueeze(0).to(device)  # [1, F, C, H, W]
    
    # 提取特征
    with torch.no_grad():
        features = encode_image(clip_model, frames, use_projection=False)  # [1, F, C]
    
    return features.squeeze(0).cpu()  # [F, C]


def extract_video_synchformer_features(video_path, synchformer_model, device, audio_length=10.0):
    """从video提取Synchformer特征"""
    # 创建临时数据集
    dataset = VideoDataset([Path(video_path)], duration_sec=audio_length)
    data = dataset.sample(0)
    
    sync_video = data['sync_video'].unsqueeze(0).to(device)  # [1, T, C, H, W]
    
    # 提取特征
    with torch.no_grad():
        sync_features = encode_video_with_sync(synchformer_model, sync_video)  # [1, S, T, D]
    
    return sync_features.squeeze(0).cpu()  # [S, T, D]


print("Feature extraction functions defined!")




from moviepy import VideoFileClip, AudioFileClip
import numpy as np

def infer_audio_speech(
    video_path,
    transcript,
    speech_start_sec,
    speech_end_sec,
    ref_audio_path,
    output_dir="./inference_output",
    gen_num=1,
    seed=0,
    device=None
):
    """
    推理函数：生成audio和speech
    
    Args:
        video_path: 视频文件路径
        transcript: 文本转录
        ref_audio_path: 参考音频路径（用于speaker embedding）
        output_dir: 输出目录
        device: 设备（默认使用CONFIG中的device）
    """
    if device is None:
        device = CONFIG["device"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Processing video: {video_path}")
    print(f"Transcript: {transcript}")
    print(f"Reference audio: {ref_audio_path}")
    
    # ========== 1. 提取所有特征 ==========
    print("\n[1/5] Extracting phoneme features...")
    if transcript is not None:
        phone_id, phone_seq = text_to_phoneme_ids(
            transcript,
            CONFIG["token_list_path"],
            speech_start_sec = speech_start_sec, 
            speech_end_sec = speech_end_sec,
            audio_duration_sec = CONFIG["audio_duration_sec"], 
            audio_length_per_sec = CONFIG["audio_length_per_sec"]
        )
        phone_id = phone_id.unsqueeze(0).to(torch.int).to(device)  # [1, latent_length]
    else:
        phone_id = torch.zeros([1,CONFIG["audio_duration_sec"] * CONFIG["audio_length_per_sec"] ]).to(torch.int).to(device)
    print(f"Phone ID shape: {phone_id.shape}")

        
    print("\n[2/5] Extracting reference audio embedding...")
    if ref_audio_path is not None:
        ref_audio_ebd = extract_ref_audio_embedding(
            ref_audio_path,
            rawnet_model,
            device
        )
        ref_audio_ebd = torch.tensor(ref_audio_ebd, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 256]
    else:
        ref_audio_ebd = torch.zeros([1, 256]).to(device)
    print(f"Reference audio embedding shape: {ref_audio_ebd.shape}")

    
    print("\n[3/5] Extracting CLIP video features...")
    video_feat = extract_video_clip_features(
        video_path,
        clip_model,
        device,
        VIDEO_PROCESS_CONFIG
    )
    video_feat = video_feat.unsqueeze(0).to(device)  # [1, F, C]
    print(f"Video CLIP features shape: {video_feat.shape}")
    
    print("\n[4/5] Extracting Synchformer features...")
    synch_feature = extract_video_synchformer_features(
        video_path,
        synchformer_model,
        device,
        CONFIG["audio_duration_sec"]
    )
    synch_feature = synch_feature.unsqueeze(0).to(device)  # [1, S, T, D]
    print(f"Synchformer features shape: {synch_feature.shape}")
    
    # ========== 2. 准备模型输入 ==========
    print("\n[5/5] Preparing model inputs...")
    batch_size = 1
    latent_length = int(CONFIG["audio_duration_sec"] * CONFIG["audio_length_per_sec"])
    
    # 处理video features
    video_feat = torch.nn.functional.interpolate(
        video_feat.permute(0, 2, 1),
        size=latent_length,
        mode='nearest'
    )  # [1, 768, latent_length]
    video_feat = video_feat.transpose(1, 2)  # [1, latent_length, 768]
    
    # 处理synchformer features
    synch_feature = synch_feature.reshape([batch_size, -1, synch_feature.shape[-1]])  # [1, S*T, D]
    synch_feature = synch_feature.transpose(1, 2)  # [1, D, S*T]
    synch_feature = torch.nn.functional.interpolate(
        synch_feature,
        size=latent_length,
        mode='nearest-exact'
    )  # [1, D, latent_length]
    synch_feature = synch_feature.transpose(1, 2)  # [1, latent_length, D]
    
    # 处理lip features (如果没有，使用零填充)
    lip_feature = torch.zeros([batch_size, latent_length, 1024], device=device)
    
    # 投影特征
    video_feat_cond = vaflow_model.cond_proj(video_feat)  # [1, latent_length, 768]
    ref_speech_cond = vaflow_model.ref_proj(ref_audio_ebd.unsqueeze(1))  # [1, 1, 768]
    phone_latent = vaflow_model.phone_embedding(phone_id)  # [1, latent_length, phone_ebd_dim]
    synch_feature = vaflow_model.synch_proj(synch_feature)  # [1, latent_length, synch_feat_dim]
    lip_feature = vaflow_model.lip_proj(lip_feature)  # [1, latent_length, lip_feat_dim]
    
    # 拼接条件特征
    video_feat_cond     = torch.concat([ref_speech_cond, video_feat_cond], dim = 1)   # [1, 1+latent_length, 768]
    cross_attn_cond     = video_feat_cond                                             # [1, 1+latent_length, 768]
    cross_attn_cond_uncond = torch.zeros_like(cross_attn_cond)                        # [1, 1+latent_length, 768]
    phone_latent = phone_latent.transpose(1, 2)  # [1, phone_ebd_dim, latent_length]
    synch_feature = synch_feature.transpose(1, 2)  # [1, synch_feat_dim, latent_length]
    lip_feature = lip_feature.transpose(1, 2)  # [1, lip_feat_dim, latent_length]


    # print(video_feat_cond.shape, phone_latent.shape, lip_feature.shape, synch_feature.shape)
    latent_cond = torch.concat(
        [lip_feature, phone_latent, synch_feature],
        dim=1
    )  # [1, 768 + phone_ebd_dim + lip_feat_dim + synch_feat_dim, latent_length]
    latent_uncond = torch.zeros_like(latent_cond)
    
    # ========== 3. 模型推理 ==========
    print("\nRunning inference...")
    wrapped_vaflow = WrappedModel(vaflow_model.vaflow)
    solver = ODESolver(velocity_model=wrapped_vaflow)
    
    for i in range(gen_num):
        generator = torch.Generator(device=device).manual_seed(seed + i)
        video_latent = torch.randn(
            (batch_size, CONFIG["original_channel"], latent_length),
            device=device,
            generator=generator
        )  # [1, original_channel, latent_length]
        video_latent = torch.cat([video_latent, latent_cond], dim=1).detach()  # [1, original_channel + cond_dim, latent_length]
        
        # Rotary embedding
        rotary_embedding = get_1d_rotary_pos_embed(
            vaflow_model.rotary_embed_dim,
            video_latent.shape[2] + 1,
            use_real=True,
            repeat_interleave_real=False,
        )
        
        # Flow matching采样
        time_grid = torch.linspace(0, 1, CONFIG["sample_steps"] + 1).to(device)
        with torch.no_grad():
            synthetic_samples = solver.sample(
                time_grid=time_grid,
                x_init=video_latent,
                method=CONFIG["sample_method"],
                return_intermediates=False,
                atol=1e-5,
                rtol=1e-5,
                step_size=None,
                global_cond=ref_speech_cond,
                latent_uncond=latent_uncond,
                w=CONFIG["guidance_scale"],
                c=cross_attn_cond,
                cross_attn_uncond=cross_attn_cond_uncond,
                rotary_embedding=rotary_embedding,
            )
        
        # ========== 4. 解码生成音频 ==========
        print("\nDecoding audio...")
        audio_latent = synthetic_samples  # [1, original_channel + cond_dim, latent_length]
        audio_latent = audio_latent[:, :CONFIG["original_channel"], :]  # [1, original_channel, latent_length]
        audio_latent = audio_latent / CONFIG["scale_factor"]
        
        # 重塑为VAE输入格式
        audio_latent = audio_latent.reshape([batch_size, -1, 8, audio_latent.shape[-1]])  # [1, 16, 8, latent_length]
        audio_latent = audio_latent.transpose(-2, -3).transpose(-2, -1)  # [1, 8, latent_length, 16]
        
        # VAE解码
        with torch.no_grad():
            mel_spectrogram = vaflow_model.vae.decode(audio_latent).sample  # [1, 1, mel_length, 64]
            gen_audio = vaflow_model.vocoder(mel_spectrogram.squeeze(1))  # [1, audio_length]
        
        gen_audio = gen_audio.cpu()
        
        # ========== 5. 保存结果 ==========
        video_clip = VideoFileClip(video_path)
        video_id = Path(video_path).stem
        audio_path = os.path.join(output_dir, f"{video_id}_generated_{seed+i}.wav")
        output_video_path = os.path.join(output_dir, f"{video_id}_generated_{seed+i}.mp4")

        torchaudio.save(audio_path, gen_audio, CONFIG["audio_sample_rate"])
        new_audio_clip = AudioFileClip(audio_path)
        video_clip.audio = new_audio_clip
        video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
        video_clip.close()
        new_audio_clip.close()

    
    # return audio_path, gen_audio

print("Inference function defined!")




gen_num = 1
output_dir = "./infer/yihan"
ref_audio_path = None
os.makedirs(output_dir, exist_ok = True)

import csv

with open('/home/chengxin/chengxin/vssflow/data/av_metadata_500.tsv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        id, video_path, transcript, speech_start_sec, speech_end_sec = row
        transcript = transcript.upper()
        speech_start_sec = float(speech_start_sec)
        speech_end_sec = float(speech_end_sec)
        
        try:
            infer_audio_speech(
                video_path=video_path,
                transcript=transcript,
                speech_start_sec=speech_start_sec,
                speech_end_sec=speech_end_sec,
                ref_audio_path=ref_audio_path,
                output_dir=output_dir,
                gen_num=gen_num,
                seed=52
            )
        except Exception as e:
            print(id, e)
            continue
        
        