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

    rand_start_sec = [i['rand_start_sec'] for i in batch]
    mix_mode = [i['mix_mode'] for i in batch]
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
            "rand_start_sec": rand_start_sec,
            "mix_mode": mix_mode,

    }



class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 # Dataset params
                 meta_dir,
                 audio_split,
                 speech_split,
                 joint_split,
                 synthetic_ratio,
                 load_mode_item : str = "video_frame", 
                 verbose        : bool = True,
                 dilimeter      : str = " ",
                 mix_modes      : list = ['add', 'insert'],   # MODIFIED
                 # Process params 
                 audio_process_config : dict = None,
                 feat_process_config  : dict = None,
                 ):
        # Dataset params
        self.load_mode_item = load_mode_item
        # if isinstance(split, str):
        #     split = [split]
        self.meta_dir       = meta_dir
        self.audio_split    = audio_split
        self.speech_split   = speech_split
        self.joint_split    = joint_split
        self.synthetic_ratio  = synthetic_ratio
        self.dilimeter        = dilimeter
        self.audio_metas      = self._load_meta(meta_dir, [audio_split])  if audio_split  is not None else None
        self.speech_metas     = self._load_meta(meta_dir, [speech_split]) if speech_split is not None else None
        self.joint_metas      = self._load_meta(meta_dir, [joint_split])  if joint_split  is not None else None
        self.mix_modes        = mix_modes

        # Audio and video relate params
        self.audio_process_config = audio_process_config
        self.duration = self.audio_process_config.duration
        self.audio_latent_fps = self.audio_process_config.audio_latent_fps
        self.channel_num = self.audio_process_config.channel_num

        # feat_config
        self.video_feat_fps = feat_process_config.video_feat_fps
        self.synch_feat_fps = feat_process_config.synch_feat_fps
        self.lip_feat_fps   = feat_process_config.lip_feat_fps



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
        if self.joint_metas is not None:
            return len(self.joint_metas)
        elif self.speech_metas is not None:
            return len(self.speech_metas)    
        else:
            raise  


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
            index = random.randint(0, self.__len__() - 1)
            return self.__getitem__(index)

# MIX 改进：
## - phoneid 和 lip_feat 在时序上偏移，但是充满说话的时间段
## - phoneid = 0 表示不提供speech的信息
## - lip_feat 用特殊标记代替，用来替代唇形不知道的情况

    def getitem_video_feat_ref_lip_synch_text_with_waveform(self, idx):
        if random.random() > self.synthetic_ratio:        
            cur_va_path, cur_feat_path, ref_audio_ebd_path, lip_feat_path, synch_feat_path, phone_id, phone_seq = self.joint_metas[idx]

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

            rand_start_sec = -1
            mix_mode = 'None'


        else:
            idx = random.randint(0, len(self.speech_metas) - 1)
            idx2 = random.randint(0, len(self.audio_metas) - 1)
            cur_va_path_s, cur_feat_path_s, ref_audio_ebd_path_s, lip_feat_path_s, synch_feat_path_s, phone_id_s, phone_seq_s = self.speech_metas[idx]
            cur_va_path_a, cur_feat_path_a, _                   , _              , synch_feat_path_a, _         , _           = self.audio_metas[idx2]

            ########## Audio Waveform ##########
            cur_video_id_a, _ = os.path.splitext(os.path.basename(cur_va_path_a))
            audio_waveform, audio_sr, audio_duration = self.prepare_audio_data_from_va_file(va_path=cur_va_path_a, return_duration=True, channel_num = self.channel_num)
            cur_video_id_s, _ = os.path.splitext(os.path.basename(cur_va_path_s))
            speech_waveform, audio_sr, speech_duration = self.prepare_audio_data_from_va_file(va_path=cur_va_path_s, return_duration=True, channel_num = self.channel_num)

            video_feat_a = np.load(cur_feat_path_a)
            video_feat_a = torch.Tensor(video_feat_a)
            video_feat_s = np.load(cur_feat_path_s)
            video_feat_s = torch.Tensor(video_feat_s)

            ref_audio_ebd_s = torch.Tensor(np.load(ref_audio_ebd_path_s))

            synch_feat_s = np.load(synch_feat_path_s)
            synch_feat_s = torch.Tensor(synch_feat_s)
            synch_feat_a = np.load(synch_feat_path_a)
            synch_feat_a = torch.Tensor(synch_feat_a)

            phone_id_s = torch.Tensor([int(i) for i in phone_id_s.split(" ")]).to(torch.int)
            phone_id_s = self.match_sequence_length(phone_id_s, int(speech_duration*self.audio_latent_fps))
            phone_seq_s = [i for i in phone_seq_s.split(" ")]
            lip_feat_s = torch.Tensor(np.load(lip_feat_path_s))

            
            rand_start_sec = random.uniform(0, audio_duration - speech_duration)
            if audio_duration < speech_duration:
                rand_start_sec = 0
                audio_duration = speech_duration
            # ## MODIFIED: rand_start_sec设置为0
            # rand_start_sec = 0

            cur_video_id       = f"speech_{cur_video_id_s}_audio_{cur_video_id_a}"
            cur_va_path     = f"speech_{cur_va_path_s}_audio_{cur_va_path_a}"
            ref_audio_ebd_path = ref_audio_ebd_path_s
            ref_audio_ebd      = ref_audio_ebd_s
            phone_seq      = phone_seq_s


            # MIX
            mix_mode = random.choice(self.mix_modes)
            feat_start_pos  = int(rand_start_sec * self.video_feat_fps)
            wave_start_pos  = int(rand_start_sec * audio_sr)
            synch_start_pos = int(rand_start_sec * self.synch_feat_fps)

            if mix_mode == 'add':
                video_feat = video_feat_a.clone()
                actual_feat_len = min(video_feat_s.shape[0], video_feat.shape[0] - feat_start_pos)
                video_feat[feat_start_pos : feat_start_pos + actual_feat_len] += video_feat_s[:actual_feat_len]

                audio_waveform = audio_waveform.clone()
                actual_wave_len = min(int(speech_duration * audio_sr), audio_waveform.shape[1] - wave_start_pos)
                audio_waveform[:, wave_start_pos : wave_start_pos + actual_wave_len] *= random.uniform(0.3, 0.7)    # MODIFIED
                audio_waveform[:, wave_start_pos : wave_start_pos + actual_wave_len] += speech_waveform[:, :actual_wave_len]
                
                synch_feat = synch_feat_a.clone()
                synch_feat   = synch_feat.reshape([-1, synch_feat.shape[-1]])
                synch_feat_s = synch_feat_s.reshape([-1, synch_feat_s.shape[-1]])                                
                actual_synch_len = min(synch_feat_s.shape[0], synch_feat.shape[0] - synch_start_pos)     
                synch_feat[synch_start_pos : synch_start_pos + actual_synch_len] += synch_feat_s[:actual_synch_len]
                synch_feat   = synch_feat.reshape([30, -1, synch_feat.shape[-1]])

            elif mix_mode == 'insert':
                video_feat = video_feat_a.clone()
                actual_feat_len = min(video_feat_s.shape[0], video_feat.shape[0] - feat_start_pos)
                video_feat[feat_start_pos : feat_start_pos + actual_feat_len] = video_feat_s[:actual_feat_len]

                audio_waveform = audio_waveform.clone()
                actual_wave_len = min(int(speech_duration * audio_sr), audio_waveform.shape[1] - wave_start_pos)
                audio_waveform[:, wave_start_pos : wave_start_pos + actual_wave_len] = speech_waveform[:, :actual_wave_len]
                
                synch_feat = synch_feat_a.clone()
                synch_feat   = synch_feat.reshape([-1, synch_feat.shape[-1]])        
                synch_feat_s = synch_feat_s.reshape([-1, synch_feat_s.shape[-1]])                          
                actual_synch_len = min(synch_feat_s.shape[0], synch_feat.shape[0] - synch_start_pos)   
                synch_feat[synch_start_pos : synch_start_pos + actual_synch_len] = synch_feat_s[:actual_synch_len]
                synch_feat   = synch_feat.reshape([30, -1, synch_feat.shape[-1]])

            else:
                raise
            
            phone_start_pos  = int(rand_start_sec * self.audio_latent_fps)
            padded_phone_id = torch.zeros(phone_start_pos, dtype=torch.int)
            phone_id         = torch.cat([padded_phone_id, phone_id_s], dim=0)    

            lip_start_pos   = int(rand_start_sec * self.lip_feat_fps)
            padded_lip_feat = torch.zeros((lip_start_pos, lip_feat_s.shape[-1]), dtype=lip_feat_s.dtype)
            lip_feat = torch.cat([padded_lip_feat, lip_feat_s], dim = 0)
            


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

            "rand_start_sec":rand_start_sec,
            "mix_mode":mix_mode

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




