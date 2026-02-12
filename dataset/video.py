import sys, os, random, traceback, librosa, decord, math
import torch.nn.functional as nn_func
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from typing import Union, List
from librosa.filters import mel as librosa_mel
from util.mel_filter import extract_batch_mel
from scipy.interpolate import interp1d
import torch, torchaudio, torchvision
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import logging
log = logging.getLogger()

# AUDIO_LATENT_FPS = 25
# AUDIO_DURATION = 10


def collate_fn_for_lip(batch):
    video_id = [i['video_id'] for i in batch]
    video_feat = torch.stack([i['video_feat'] for i in batch])
    video_path = [i['video_path'] for i in batch]

    ref_audio_path = [i['ref_audio_path'] for i in batch]
    ref_audio_ebd = torch.stack([i['ref_audio_ebd'] for i in batch])

    synch_feat = torch.stack([i['synch_feat'] for i in batch])

    length_l = max([i['lip_feat'].shape[0] for i in batch])
    length_p = max([i['phone_id'].shape[-1] for i in batch])
    lip_feat = torch.stack([torch.nn.functional.pad(i['lip_feat'], (0, 0, 0, length_l - i['lip_feat'].shape[0]), mode='constant', value=0.0) for i in batch])
    phone_id = torch.stack([torch.nn.functional.pad(i['phone_id'], (0,       length_p - i['phone_id'].shape[-1]), mode='constant', value=0) for i in batch])
    phone_seq = [i['phone_seq'] for i in batch]

    audio_waveform = torch.stack([i['audio_waveform'] for i in batch])
    audio_sr = [i['audio_sr'] for i in batch]
    audio_duration = [i['audio_duration'] for i in batch]


    return {
            "video_id": video_id,
            "video_feat": video_feat,
            "video_path": video_path,
            "ref_audio_path": ref_audio_path,
            "ref_audio_ebd": ref_audio_ebd,
            "synch_feat": synch_feat,
            "lip_feat": lip_feat,
            "phone_id": phone_id,
            "phone_seq": phone_seq,
            "audio_waveform": audio_waveform,
            "audio_sr": audio_sr,
            "audio_duration": audio_duration,

    }



class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 # Dataset params
                 meta_dir   : str,
                 split      : Union[str, List[str]],
                 load_mode_item : str = "video_frame", 
                 verbose        : bool = True,
                 dilimeter      : str = " ",
                 # Process params 
                 audio_process_config : dict = None,
                 video_process_config : dict = None,
                 ):
        # Dataset params
        self.load_mode_item = load_mode_item
        if isinstance(split, str):
            split = [split]
        self.meta_dir   = meta_dir
        self.split      = split
        self.dilimeter  = dilimeter
        self.metas      = self._load_meta(meta_dir, split)
        
        # Audio and video relate params
        self.audio_process_config = audio_process_config
        self.duration = self.audio_process_config.duration
        self.audio_latent_fps = self.audio_process_config.audio_latent_fps
        self.channel_num = self.audio_process_config.channel_num



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
            if  self.load_mode_item == 'video_feat_ref_lip_synch_text_with_waveform':
                return self.getitem_video_feat_ref_lip_synch_text_with_waveform(idx)
            else:
                raise NotImplementedError(f"Load mode `{self.load_mode_item}` is not implemented.")
        except KeyboardInterrupt as e:
            raise e
        except NotImplementedError as e:
            raise e
        except BaseException as e:
            log.info("Failed to load video item with error `{}`, auto replace with a random item.".format(e))
            print(traceback.format_exc())
            index = random.randint(0, len(self.metas) - 1)
            return self.__getitem__(index)

# MIX 改进：
## - phoneid 和 lip_feat 在时序上偏移，但是充满说话的时间段
## - phoneid = 0 表示不提供speech的信息
## - lip_feat 用特殊标记代替，用来替代唇形不知道的情况

    def getitem_video_feat_ref_lip_synch_text_with_waveform(self, idx):
        cur_va_path, cur_feat_path, ref_audio_ebd_path, lip_feat_path, synch_feat_path, phone_id, phone_seq = self.metas[idx]

        ########## Audio Waveform ##########
        cur_video_id, _ = os.path.splitext(os.path.basename(cur_va_path))
        audio_waveform, audio_sr, audio_duration = self.prepare_audio_data_from_va_file(va_path=cur_va_path, return_duration=True, channel_num = self.channel_num)


        ########## Video Feat Path, CAVP / CLIP, 10 FPS * 10 s ##########
        if '.npz' in cur_feat_path:                 ## CAVP
            with np.load(cur_feat_path) as data:
                video_feat = data['feat']
            video_feat = self.align_to_target(video_feat, 100)
            video_feat = torch.from_numpy(video_feat)
        else:                                       ## CLIP
            video_feat = np.load(cur_feat_path)
        video_feat = torch.Tensor(video_feat)


        ########## Rawnet Reference Audio EBD ##########
        ref_audio_ebd = torch.Tensor(np.load(ref_audio_ebd_path))


        ########## Synchformer Featur Path ##########
        synch_feat = np.load(synch_feat_path)
        synch_feat = torch.Tensor(synch_feat)


        ########### Phoneme Id Sequence ##########
        phone_id = torch.Tensor([int(i) for i in phone_id.split(" ")]).to(torch.int)
        phone_id = self.match_sequence_length(phone_id, int(audio_duration*self.audio_latent_fps))
        phone_seq = [i for i in phone_seq.split(" ")]
        lip_feat = torch.Tensor(np.load(lip_feat_path))

        return {
            "video_id": cur_video_id,
            "video_path": cur_va_path,
            "video_feat": video_feat,
            
            "ref_audio_path": ref_audio_ebd_path,
            "ref_audio_ebd": ref_audio_ebd,

            "lip_feat": lip_feat,
            "synch_feat": synch_feat,

            "phone_id": phone_id,
            "phone_seq": phone_seq,

            "audio_waveform": audio_waveform,
            "audio_sr": audio_sr,
            "audio_duration": audio_duration,

        }


    def align_to_target(self, 
                    arr, 
                    target_frames=100, 
                    target_dim=768, 
                    kind='nearest'):

        N, D = arr.shape
        if N == target_frames:
            interpolated = arr.copy()
        else:
            src_times = np.linspace(0, target_frames - 1, N)
            dst_times = np.arange(target_frames)
            interpolator = interp1d(src_times, arr, kind=kind, axis=0, bounds_error=False, fill_value="extrapolate")
            interpolated = interpolator(dst_times)      # shape: [target_frames, D]

        if D >= target_dim:
            padded = interpolated[:, :target_dim]
        else:
            pad_width = target_dim - D
            padded = np.pad(interpolated, pad_width=((0, 0), (0, pad_width)),mode='constant', constant_values=0)
        
        return padded.astype(np.float32)  


    def match_sequence_length(self, x: torch.Tensor, target_len: int, downsample_mode: str = "nearest") -> torch.Tensor:
        """
        Resize a tensor from [l] to [target_len].
        - If l < target_len: use repeat_interleave logic (no truncation).
        - If l > target_len: use interpolation-based downsample.
        """
        cur_len = x.shape[0]
        if cur_len < target_len:
            repeat_factor = math.ceil(target_len / cur_len)
            x = x.repeat_interleave(repeat_factor, dim=0)   

        return nn_func.interpolate(x[None, None, :].float(), size=target_len, mode=downsample_mode).squeeze().long()


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




