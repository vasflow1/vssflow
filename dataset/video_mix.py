import sys, os, random, traceback, librosa, decord, math
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange
from typing import Union, List
from librosa.filters import mel as librosa_mel
from util.mel_filter import extract_batch_mel

import torch, torchaudio, torchvision
import torch.nn.functional as F
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
    mix_method = [i['mix_method'] for i in batch]

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
            "phone_seq": phone_seq,
            "mix_method": mix_method
    }



class VideoDataset_MixAudioSpeech(torch.utils.data.Dataset):
    def __init__(self, 
                 # Dataset params
                 meta_dir       : str,
                 audio_split    : Union[str, List[str]],
                 speech_split   : Union[str, List[str]],
                 load_mode_item : str = "video_frame", 
                 verbose        : bool = True,
                 dilimeter      : str = " ",
                 # Mix params
                 mix_methods    : list = ['add', 'add_novideo', 'insert'],
                 mix_prob       : float = 0.5,
                 mix_scale      : list[float] = [0.3, 0.6],
                 # Process params 
                 audio_process_config : dict = None,
                 ):
        # Dataset params
        self.meta_dir = meta_dir
        self.load_mode_item = load_mode_item
        self.verbose = verbose
        if isinstance(audio_split, str): audio_split = [audio_split]
        if isinstance(speech_split, str): speech_split = [speech_split]
        self.audio_split      = audio_split
        self.speech_split     = speech_split
        self.dilimeter  = dilimeter
        self.audio_metas      = self._load_meta(meta_dir, audio_split)
        self.speech_metas      = self._load_meta(meta_dir, speech_split)
        self.mix_methods = mix_methods
        self.mix_prob = mix_prob
        self.mix_scale = mix_scale
        
        # Audio and video relate params
        self.audio_process_config = audio_process_config
        self.duration = self.audio_process_config.duration
        self.channel_num = self.audio_process_config.channel_num


    def _load_meta(self, meta_dir, split, ext=".csv", meta_item_num=-1):
        if split is None or len(split) == 0 or split[0] == "":
            raise NotImplementedError(f"Split is empty.")
        else:
            cur_metas = []
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

        assert len(cur_metas) > 0, f"Meta data is empty."
        return cur_metas


    def __len__(self):
        return len(self.audio_metas)
        

    def __getitem__(self, idx,):
        try:
            if self.load_mode_item == "mixed_video_feat_ref_text_with_waveform":
                return self.getitem_mixed_video_feat_ref_text_with_waveform(idx)
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
            index = random.randint(0, len(self.audio_metas) - 1)
            return self.__getitem__(index)
        
    def getitem_mixed_video_feat_ref_text_with_waveform(self, idx):
        audio_path, audio_feat_path, ref_audio_path, duration_matrix, phone_id, phone_seq = self.audio_metas[idx]
        cur_audio_id, _ = os.path.splitext(os.path.basename(audio_path))
        cur_video_id = f"audio_{cur_audio_id}"
        audio_video_feat = torch.Tensor(np.load(audio_feat_path))
        ref_audio_ebd = torch.Tensor(np.load(ref_audio_path))
        audio_waveform, audio_sr, audio_duration = self.prepare_audio_data_from_va_file(va_path=audio_path, return_duration=True, channel_num = self.channel_num)
        phone_id = torch.Tensor([int(i) for i in phone_id.split(" ")]).to(torch.int)
        phone_seq = [i for i in phone_seq.split(" ")]
        duration_matrix = torch.Tensor(np.load(duration_matrix))
        mix_method = 'None'


        if random.uniform(0, 1) < self.mix_prob:
            idx = random.randint(0, len(self.speech_metas) - 1)
            speech_path, speech_feat_path, ref_audio_path, duration_matrix, phone_id, phone_seq = self.speech_metas[idx] #  ref_audio_path, duration_matrix, phone_id, phone_seq 重载一下
            cur_speech_id, _ = os.path.splitext(os.path.basename(speech_path))
            speech_video_feat = torch.Tensor(np.load(speech_feat_path))
            ref_audio_ebd = torch.Tensor(np.load(ref_audio_path))
            speech_waveform, speech_sr, speech_duration = self.prepare_audio_data_from_va_file(va_path=speech_path, return_duration=True, channel_num = self.channel_num)
            phone_id = torch.Tensor([int(i) for i in phone_id.split(" ")]).to(torch.int)
            phone_seq = [i for i in phone_seq.split(" ")]
            duration_matrix = torch.Tensor(np.load(duration_matrix))

            audio_video_feat, audio_waveform, speech_start, mix_method  = self.random_mix(audio_video_feat, audio_waveform, audio_duration,
                                                                                         speech_video_feat, speech_waveform, speech_duration)
            phone_id, duration_matrix = self.adjust_speech(speech_start, phone_id, duration_matrix)
            cur_video_id = f"audio_{cur_audio_id}_speech_{cur_speech_id}_{mix_method}_{speech_start}-{round(speech_duration+speech_start,2)}"


        return {
            "video_id": cur_video_id,
            "video_feat": audio_video_feat,
            "video_path": audio_waveform,
            "ref_audio_ebd": ref_audio_ebd,
            "ref_audio_path": ref_audio_path,
            "audio_waveform": audio_waveform,
            "audio_sr": audio_sr,
            "audio_duration": audio_duration,
            "duration_matrix": duration_matrix,
            "phone_id": phone_id,
            "phone_seq": phone_seq,
            "mix_method": mix_method,
        }


        # cur_video_id, _ = os.path.splitext(os.path.basename(cur_va_path))
        # video_feat = torch.Tensor(np.load(cur_feat_path))
        # ref_audio_ebd = torch.Tensor(np.load(ref_audio_path))
    
        # audio_waveform, audio_sr, audio_duration = self.prepare_audio_data_from_va_file(va_path=cur_va_path, return_duration=True, channel_num = self.channel_num)
        # if 'audio_fake_duration' in duration_matrix:
        #     cur_video_id = f"audio_{cur_video_id}"
        #     # phone_id = torch.Tensor([0]).to(torch.int)
        #     # phone_seq = ["<blank>"]
        #     # duration_matrix = torch.zeros([self.latent_length, 1])
        # else:
        #     cur_video_id = f"speech_{cur_video_id}"

        # phone_id = torch.Tensor([int(i) for i in phone_id.split(" ")]).to(torch.int)
        # phone_seq = [i for i in phone_seq.split(" ")]
        # duration_matrix = torch.Tensor(np.load(duration_matrix))

        # return {
        #     "video_id": cur_video_id,
        #     "video_feat": video_feat,
        #     "video_path": cur_va_path,
        #     "ref_audio_ebd": ref_audio_ebd,
        #     "ref_audio_path": ref_audio_path,
        #     "audio_waveform": audio_waveform,
        #     "audio_sr": audio_sr,
        #     "audio_duration": audio_duration,
        #     "duration_matrix": duration_matrix,
        #     "phone_id": phone_id,
        #     "phone_seq": phone_seq
        # }


    def adjust_speech(self, speech_start_duration, phone_id, duration_matrix):
        # phone_id [phone_length]    duration_matrix [250, phone_length]
        # phone_id = torch.cat((torch.tensor([59]).to(phone_id), phone_id))

        assert phone_id[0] == 59
        phone_id = torch.cat((torch.tensor([0], dtype=torch.int).to(phone_id), phone_id))            # [1 + phone_length]
        silence_length = int(round(speech_start_duration*25, 0))
        duration_matrix_new = torch.zeros([silence_length, phone_id.shape[-1]])     # [append_dur_length, 1 + phone_length]
        duration_matrix_new[:, 0] = 1

        assert duration_matrix[-silence_length].sum() == 0
        duration_matrix = duration_matrix[:-silence_length]                              # [250 - append_dur_length, phone_length]
        duration_matrix = F.pad(duration_matrix, (1, 0), mode='constant', value=0)       # [250 - append_dur_length, 1 + phone_length]
        duration_matrix_new = torch.cat((duration_matrix_new, duration_matrix), dim=0)   # [250, 1 + phone_length]
        return phone_id, duration_matrix_new
    

    def random_mix(self, audio_video_feat, audio_waveform, audio_duration, speech_video_feat, speech_waveform, speech_duration):
        
        mix_method = random.choice(self.mix_methods)
        speech_start = round(random.uniform(0, audio_duration - speech_duration), 2)
        scale = random.uniform(self.mix_scale[0], self.mix_scale[1])

        wave_start_pos = int(speech_start * self.audio_process_config.sample_rate)
        wave_speech_length = int(speech_duration * self.audio_process_config.sample_rate)

        video_feat_fps = 10 #
        feat_start_pos = int(speech_start * video_feat_fps)
        feat_speech_length = int(speech_duration * video_feat_fps)

        # add
        if 'add' in mix_method:
            audio_waveform = audio_waveform * scale
            audio_waveform[:, wave_start_pos : wave_start_pos + wave_speech_length] = audio_waveform[:, wave_start_pos : wave_start_pos + wave_speech_length] + speech_waveform[:,:wave_speech_length]
            if 'novideo' not in mix_method:
                audio_video_feat[feat_start_pos : feat_start_pos + feat_speech_length] = audio_video_feat[feat_start_pos : feat_start_pos + feat_speech_length] + speech_video_feat[:feat_speech_length]


        # insert, 是不是也可以不在video_feat上加？
        elif 'insert' in mix_method:
            audio_video_feat[feat_start_pos : feat_start_pos + feat_speech_length] = speech_video_feat[:feat_speech_length]
            audio_waveform[:, wave_start_pos : wave_start_pos + wave_speech_length] = speech_waveform[:,:wave_speech_length]
            audio_video_feat[feat_start_pos : feat_start_pos + feat_speech_length] = speech_video_feat[:feat_speech_length]

        else:
            raise NotImplementedError(f"Invalid mix method: {mix_method}" )



        # insert
        
        return audio_video_feat, audio_waveform, speech_start, mix_method
        

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
        
        return audio_waveform, audio_sr, audio_duration
        # if return_duration:
        #     return audio_waveform, audio_sr, audio_duration
        # else:
        #     return audio_waveform, audio_sr





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