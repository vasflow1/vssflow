import sys, os, random, traceback, librosa, decord, math
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from typing import Union, List
from librosa.filters import mel as librosa_mel
from util.mel_filter import extract_batch_mel

import torch, torchaudio, torchvision
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



def collate_fn_for_text(batch):
    video_id = [i['video_id'] for i in batch]
    video_feat = torch.stack([i['video_feat'] for i in batch])
    video_path = [i['video_path'] for i in batch]
    ref_audio_ebd = torch.stack([i['ref_audio_ebd'] for i in batch])
    ref_audio_path = [i['ref_audio_path'] for i in batch]
    audio_waveform = torch.stack([i['audio_waveform'] for i in batch])
    audio_sr = [i['audio_sr'] for i in batch]
    audio_duration = [i['audio_duration'] for i in batch]

    length = max([len(i['phone_id']) for i in batch])
    duration_matrix = torch.stack([torch.nn.functional.pad(i['duration_matrix'], (0, length - i['duration_matrix'].shape[-1]), mode='constant', value=0) for i in batch])
    phone_id = torch.stack([torch.nn.functional.pad(i['phone_id'], (0, length - i['phone_id'].shape[-1]), mode='constant', value=0) for i in batch])
    phone_seq = [i['phone_seq'] for i in batch]
    return {
            "video_id": video_id,
            "video_feat": video_feat,
            "video_path": video_path,
            "ref_audio_ebd": ref_audio_ebd,
            "ref_audio_path": ref_audio_path,
            "audio_waveform": audio_waveform,
            "audio_sr": audio_sr,
            "audio_duration": audio_duration,
            "duration_matrix": duration_matrix,
            "phone_id": phone_id,
            "phone_seq": phone_seq
    }

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 # Dataset params
                 meta_dir   : str,
                 split      : Union[str, List[str]],
                 load_mode_meta : str = "", 
                 load_mode_item : str = "video_frame", 
                 verbose        : bool = True,
                 dilimeter      : str = " ",
                 # Process params 
                 audio_process_config : dict = None,
                 video_process_config : dict = None,
                 ):
        # Dataset params
        self.load_mode_meta = load_mode_meta
        self.load_mode_item = load_mode_item
        self.verbose = verbose
        if isinstance(split, str):
            split = [split]
        self.meta_dir   = meta_dir
        self.split      = split
        self.dilimeter  = dilimeter
        self.metas      = self._load_meta(meta_dir, split)
        
        # Audio and video relate params
        self.audio_process_config = audio_process_config
        self.video_process_config = video_process_config
        assert self.audio_process_config.duration == self.video_process_config.duration, "Audio and video duration should be the same."
        self.duration = self.audio_process_config.duration
        self.video_process_config.target_frame_length = int(self.duration * self.video_process_config.target_sampling_rate)
        self.channel_num = self.audio_process_config.channel_num

        # Video frame transform
        if isinstance(self.video_process_config.resize_input_size, int):
            video_fram_h = video_frame_w = self.video_process_config.resize_input_size
        else:
            assert len(self.video_process_config.resize_input_size) == 2, "Resize input size should be a tuple of (h, w)."
            video_fram_h, video_frame_w = self.video_process_config.resize_input_size
        self.video_input_size = [video_fram_h, video_frame_w]
        self.clip_transform = transforms.Compose([
            transforms.Resize((video_fram_h, video_frame_w), interpolation=BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def _load_meta(self, meta_dir, split, ext=".csv", meta_item_num=-1):
        cur_metas = []

        if split is None or len(split) == 0 or split[0] == "":
            raise NotImplementedError(f"Split is empty.")

        else:
            for cur_split in split:
                cur_meta_path = os.path.join(meta_dir, cur_split+ext)
                if not os.path.exists(cur_meta_path):
                    raise RuntimeError(f"Meta file {cur_meta_path} doesn't exist.")
                with open(cur_meta_path, "r") as split_f:
                    cur_data = split_f.readlines()
                    cur_data = [line.strip() for line in cur_data if line.strip() != ""]
                for line in cur_data:
                    line_info = line.split(self.dilimeter)
                    # line format: video_path/frame_path, audio_path, start_idx, total_frames, label
                    if meta_item_num >0 and len(line_info) < meta_item_num:
                        raise RuntimeError(f"Video input format is not correct, missing one or more element. Meta line: {line}")
                    
                    cur_metas.append(line_info)

        # TODO Filter out invalid meta data

        assert len(cur_metas) > 0, f"Meta data is empty."

        return cur_metas

    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx,):
        try:
            if self.load_mode_item == "video_frame":
                return self.getitem_video_frame(idx)
            elif self.load_mode_item == "video_feat":
                return self.getitem_video_feat(idx)
            elif self.load_mode_item == "video_feat_with_ib":
                return self.getitem_video_feat_with_ib(idx)
            elif self.load_mode_item == "video_feat_cavp_with_ib":
                return self.getitem_video_feat_cavp_with_ib(idx)  
            elif self.load_mode_item == "video_feat_vidtok_with_ib":
                return self.getitem_video_feat_vidtok_with_ib(idx)
            elif self.load_mode_item == "video_feat_with_waveform":
                return self.getitem_video_feat_with_waveform(idx)
            elif self.load_mode_item == "video_feat_cavp_with_waveform":
                return self.getitem_video_feat_cavp_with_waveform(idx)
            elif self.load_mode_item == "video_feat_vidtok_with_waveform":
                return self.getitem_video_feat_vidtok_with_waveform(idx)
            elif self.load_mode_item == "video_feat_text_with_waveform":
                return self.getitem_video_feat_text_with_waveform(idx)
            elif self.load_mode_item == "video_feat_ref_text_with_waveform":
                return self.getitem_video_feat_ref_text_with_waveform(idx)
            else:
                raise NotImplementedError(f"Load mode `{self.load_mode_item}` is not implemented.")
        except KeyboardInterrupt as e:
            raise e
        except NotImplementedError as e:
            raise e
        except BaseException as e:
            if self.verbose:
                print("Failed to load video item with error `{}`, auto replace with a random item.".format(e))
                print(traceback.format_exc())
            index = random.randint(0, len(self.metas) - 1)
            return self.__getitem__(index)
        
    def getitem_video_frame(self, idx):
        cur_va_path, *_ = self.metas[idx]
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_va_path))
        
        video_frame = self.prepare_video_data_from_va_file(va_path=cur_va_path,)
        
        return {
            "video_id": cur_video_id,
            "video_frame": video_frame,
            "video_path": cur_va_path,
        }
    
    def getitem_video_feat(self, idx):
        cur_va_path, cur_feat_path, *_ = self.metas[idx]
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_feat_path))
        
        video_feat = np.load(cur_feat_path)
        
        return {
            "video_id": cur_video_id,
            "video_feat": video_feat,
            "video_path": cur_va_path,
        }
    
    def getitem_video_feat_with_ib(self, idx):
        cur_va_path, cur_feat_path, cur_ib_v_path, cur_ib_a_path, *_ = self.metas[idx]
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_feat_path))
        
        video_feat = np.load(cur_feat_path)
        ibv = np.load(cur_ib_v_path)
        iba = np.load(cur_ib_a_path)
        
        return {
            "video_id": cur_video_id,
            "video_feat": video_feat,
            "video_path": cur_va_path,
            "ib_video_feat": ibv,
            "ib_audio_feat": iba,
        }
    
    def getitem_video_feat_cavp_with_ib(self, idx):
        cur_va_path, cur_feat_path, cur_ib_v_path, cur_ib_a_path, cur_cavp_v_path, *_ = self.metas[idx]
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_feat_path))
        
        video_feat = np.load(cur_cavp_v_path)["feat"]
        ibv = np.load(cur_ib_v_path)
        iba = np.load(cur_ib_a_path)
        
        return {
            "video_id": cur_video_id,
            "video_feat": video_feat,
            "video_path": cur_va_path,
            "ib_video_feat": ibv,
            "ib_audio_feat": iba,
        }

    def getitem_video_feat_vidtok_with_ib(self, idx):
        cur_va_path, cur_feat_path, cur_ib_v_path, cur_ib_a_path, cur_vidtok_path, *_ = self.metas[idx]
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_feat_path))
        
        if cur_vidtok_path.endswith(".npz"):
            cur_vidtok_path = cur_vidtok_path.replace(".npz", ".npy")
        video_feat = np.load(cur_vidtok_path)
        # Trans
        video_feat = video_feat.astype(np.float32)      # 转换为float32
        video_feat = video_feat.transpose(1, 0, 2, 3)   # 将形状变为(25, 4, 8, 8)
        video_feat = video_feat.reshape(video_feat.shape[0], -1) 

        ibv = np.load(cur_ib_v_path)
        iba = np.load(cur_ib_a_path)
        
        return {
            "video_id": cur_video_id,
            "video_feat": video_feat,
            "video_path": cur_va_path,
            "ib_video_feat": ibv,
            "ib_audio_feat": iba,
        }
        
    def getitem_video_feat_with_waveform(self, idx):
        cur_va_path, cur_feat_path, *_ = self.metas[idx]
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_feat_path))
        
        video_feat = np.load(cur_feat_path)
        
        audio_waveform, audio_sr = self.prepare_audio_data_from_va_file(va_path=cur_va_path,)
        
        return {
            "video_id": cur_video_id,
            "video_feat": video_feat,
            "video_path": cur_va_path,
            "audio_waveform": audio_waveform,
            "audio_sr": audio_sr
        }
    

    def getitem_video_feat_text_with_waveform(self, idx):
        cur_va_path, cur_feat_path, duration_matrix, phone_id, phone_seq = self.metas[idx]
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_va_path))
        video_feat = torch.Tensor(np.load(cur_feat_path))
    
        audio_waveform, audio_sr, audio_duration = self.prepare_audio_data_from_va_file(va_path=cur_va_path, return_duration=True, channel_num = self.channel_num)
        if 'audio_fake_duration' in duration_matrix:
            cur_video_id = f"audio_{cur_video_id}"
            # phone_id = torch.Tensor([0]).to(torch.int)
            # phone_seq = ["<blank>"]
            # duration_matrix = torch.zeros([self.latent_length, 1])
        else:
            cur_video_id = f"speech_{cur_video_id}"

        phone_id = torch.Tensor([int(i) for i in phone_id.split(" ")]).to(torch.int)
        phone_seq = [i for i in phone_seq.split(" ")]
        duration_matrix = torch.Tensor(np.load(duration_matrix))

        return {
            "video_id": cur_video_id,
            "video_feat": video_feat,
            "video_path": cur_va_path,
            "audio_waveform": audio_waveform,
            "audio_sr": audio_sr,
            "audio_duration": audio_duration,
            "duration_matrix": duration_matrix,
            "phone_id": phone_id,
            "phone_seq": phone_seq
        }
    

    def getitem_video_feat_ref_text_with_waveform(self, idx):
        cur_va_path, cur_feat_path, ref_audio_path, duration_matrix, phone_id, phone_seq = self.metas[idx]
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_va_path))
        video_feat = torch.Tensor(np.load(cur_feat_path))
        ref_audio_ebd = torch.Tensor(np.load(ref_audio_path))
    
        audio_waveform, audio_sr, audio_duration = self.prepare_audio_data_from_va_file(va_path=cur_va_path, return_duration=True, channel_num = self.channel_num)
        if 'audio_fake_duration' in duration_matrix:
            cur_video_id = f"audio_{cur_video_id}"
            # phone_id = torch.Tensor([0]).to(torch.int)
            # phone_seq = ["<blank>"]
            # duration_matrix = torch.zeros([self.latent_length, 1])
        else:
            cur_video_id = f"speech_{cur_video_id}"

        phone_id = torch.Tensor([int(i) for i in phone_id.split(" ")]).to(torch.int)
        phone_seq = [i for i in phone_seq.split(" ")]
        duration_matrix = torch.Tensor(np.load(duration_matrix))

        return {
            "video_id": cur_video_id,
            "video_feat": video_feat,
            "video_path": cur_va_path,
            "ref_audio_ebd": ref_audio_ebd,
            "ref_audio_path": ref_audio_path,
            "audio_waveform": audio_waveform,
            "audio_sr": audio_sr,
            "audio_duration": audio_duration,
            "duration_matrix": duration_matrix,
            "phone_id": phone_id,
            "phone_seq": phone_seq
        }


        
    def getitem_video_feat_cavp_with_waveform(self, idx):
        cur_va_path, cur_feat_path, cur_ib_v_path, cur_ib_a_path, cur_cavp_v_path, *_ = self.metas[idx]
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_feat_path))
        
        video_feat = np.load(cur_cavp_v_path)["feat"]
        
        audio_waveform, audio_sr = self.prepare_audio_data_from_va_file(va_path=cur_va_path,)
        
        return {
            "video_id": cur_video_id,
            "video_feat": video_feat,
            "video_path": cur_va_path,
            "audio_waveform": audio_waveform,
            "audio_sr": audio_sr
        }

    def getitem_video_feat_vidtok_with_waveform(self, idx):
        cur_va_path, cur_feat_path, cur_ib_v_path, cur_ib_a_path, cur_vidtok_path, *_ = self.metas[idx]
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_feat_path))
        
        if cur_vidtok_path.endswith(".npz"):
            cur_vidtok_path = cur_vidtok_path.replace(".npz", ".npy")
        video_feat = np.load(cur_vidtok_path)
        # Trans
        video_feat = video_feat.astype(np.float32)      # 转换为float32
        video_feat = video_feat.transpose(1, 0, 2, 3)   # 将形状变为(25, 4, 8, 8)
        video_feat = video_feat.reshape(video_feat.shape[0], -1) 
        
        audio_waveform, audio_sr = self.prepare_audio_data_from_va_file(va_path=cur_va_path,)
        
        return {
            "video_id": cur_video_id,
            "video_feat": video_feat,
            "video_path": cur_va_path,
            "audio_waveform": audio_waveform,
            "audio_sr": audio_sr
        }
    

    def prepare_audio_data_from_va_file(self, va_path, return_duration = False, channel_num = 2):
        try:
            audio_waveform, audio_sr = torchaudio.load(va_path)
        except BaseException as e:
            raise e
        
        audio_duration = round(audio_waveform.shape[-1]/audio_sr, 2)
        if audio_sr != self.audio_process_config.sample_rate:
            audio_waveform = torchaudio.transforms.Resample(audio_sr, self.audio_process_config.sample_rate)(audio_waveform)
            audio_sr = self.audio_process_config.sample_rate
        if channel_num == 2 and audio_waveform.size(0) == 1: # Mono audio, duplicate the channel to create stereo.
            audio_waveform = audio_waveform.repeat(2, 1)
        elif channel_num == 1 and audio_waveform.size(0) != 1:  # stereo audio, mean the channel to create mono.
            audio_waveform = audio_waveform.mean(dim = 0, keepdim=True)


        # Pad and trim audio waveform
        if audio_waveform.size(1) < int(audio_sr*self.audio_process_config.duration):
            p = int(audio_sr*self.audio_process_config.duration) - audio_waveform.size(1)
            audio_waveform = torch.cat([audio_waveform, torch.zeros(channel_num, p)], dim=1)
        elif audio_waveform.size(1) > int(audio_sr*self.audio_process_config.duration):
            audio_waveform = audio_waveform[:, :int(audio_sr*self.audio_process_config.duration)]
        
        if return_duration:
            return audio_waveform, audio_sr, audio_duration
        else:
            return audio_waveform, audio_sr


    def prepare_video_data_from_va_file(self, va_path, backend="decord",):
        ''' Load video data '''
        def check_and_drop(int_list, min_value, max_value):
            return [x for x in int_list if min_value <= x <= max_value]
            return [max(min_value, min(max_value, x)) for x in int_list]

        if backend == "torchvision":
            try:
                frame_data, _, meta = torchvision.io.read_video(va_path, pts_unit="sec", output_format="THWC")
                video_raw_fps = meta["video_fps"]
            except BaseException as e:
                raise e
            video_raw_frame_num = len(frame_data)
            video_raw_duration = video_raw_frame_num / video_raw_fps
            if video_raw_duration <= self.video_process_config.raw_duration_min_threshold:
                raise RuntimeError(f"Video duration {video_raw_duration} is too short, less than {self.video_process_config.raw_duration_min_threshold}.")
            
            # Sample frames with target sampling rate, 
            # NOTE Implemention of `FPS`` here is just selecting frames in raw 30fps video frames
            video_sampled_frame_ids = [ round(i * video_raw_fps / self.video_process_config.target_sampling_rate) 
                                        for i in range(self.video_process_config.target_frame_length) ]
            video_sampled_frame_ids = check_and_drop(video_sampled_frame_ids, 0, video_raw_frame_num-1)
            sampled_frames = frame_data[video_sampled_frame_ids, :, :, :]
            
            if sampled_frames.dtype != torch.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).byte()
                else:
                    frame = frame.byte()
            
            sampled_frames = [Image.fromarray(frame.numpy()) for frame in sampled_frames]

        elif backend == "decord":
            try:
                decord_vr = decord.VideoReader(va_path, num_threads=1, ctx=decord.cpu(0))
            except BaseException as e:
                raise e
            video_raw_frame_num = len(decord_vr)
            video_raw_fps = decord_vr.get_avg_fps()
            video_raw_duration = video_raw_frame_num / video_raw_fps
            if video_raw_duration <= self.video_process_config.raw_duration_min_threshold:
                raise RuntimeError(f"Video duration {video_raw_duration} is too short, less than {self.video_process_config.raw_duration_min_threshold}.")
            
            # Sample frames with target sampling rate, 
            # NOTE Implemention of `FPS`` here is just selecting frames in raw 30fps video frames
            video_sampled_frame_ids = [ round(i * video_raw_fps / self.video_process_config.target_sampling_rate)
                                        for i in range(self.video_process_config.target_frame_length) ]
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
        p = self.video_process_config.target_frame_length - len(sampled_frames)
        if p > 0:
            w, h = sampled_frames[0].size  
            blank_pil = Image.new('RGB', (h, w), (0, 0, 0))
            sampled_frames = sampled_frames + [blank_pil] * p
        else:
            sampled_frames = sampled_frames[0 : self.video_process_config.target_frame_length]
        assert len(sampled_frames) == self.video_process_config.target_frame_length, f"Sampled frames length {len(sampled_frames)} is not equal to target length {self.video_target_frame_len}."
        # Transform and stack
        sampled_frames = [self.clip_transform(sampled_frame) for sampled_frame in sampled_frames ]
        processed_frames = torch.stack(sampled_frames, dim=0)

        return processed_frames


if __name__=="__main__":
    from omegaconf import OmegaConf
    
    meta_dir = "/data_mount/audioset/music_sub/meta/minus_musiccaps_split"
    split    = "train"
    audio_process_config = {
        'duration': 10.0,
    }
    video_process_config = {
        'duration'                      : 10.0,
        'resize_input_size'             : 224,
        'target_sampling_rate'          : 10,
        'raw_duration_min_threshold'    : 2.0,
    }
    audio_process_config = OmegaConf.create(audio_process_config)
    video_process_config = OmegaConf.create(video_process_config)
    
    dataset = VideoDataset(
        meta_dir=meta_dir,
        split=split,
        audio_process_config=audio_process_config,
        video_process_config=video_process_config,
    )
    
    torch.manual_seed(42)
    dl = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True, num_workers=0)
    
    for c, i in tqdm(enumerate(dl), total=len(dl)):
        for k, v in i.items():
            if isinstance(v, torch.Tensor):
                print(k, v.shape,)
        if c > 100:
            break
        # print(c)
        pass