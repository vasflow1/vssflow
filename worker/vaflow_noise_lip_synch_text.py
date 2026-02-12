import os, time, random, einops, wandb, inspect
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from functools import partial
from scipy.io.wavfile import write as scipy_wav_write
from typing import Any, Callable, Dict, List, Optional, Union

import torch, torchaudio
import torch.nn as nn
import torchvision.io as tvio
import pytorch_lightning as pl
import torch.nn.functional as nn_func
from pytorch_lightning.callbacks import BaseFinetuning
from safetensors import safe_open
import copy

# from peft import LoraConfig, get_peft_model
# from peft.utils import get_peft_model_state_dict

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

from diffusers.models import AutoencoderOobleck
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.utils.torch_utils import randn_tensor

from .base import instantiate_from_config, get_class_from_config
from model.clip.clip_module import CLIPViT
from model.stable_audio.stable_audio_transformer import StableAudioDiTModel
from transformers import SpeechT5HifiGan
from diffusers.models import AutoencoderKL
from util.mel_filter import extract_batch_mel
import math



class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor, latent_uncond = None, cross_attn_uncond = None, w: float = 3.0, rotary_embedding=None, **extras):
        t = t.unsqueeze(0).repeat(x.shape[0]).to(x)

        c_uncond = cross_attn_uncond
        if cross_attn_uncond is None:
            c_uncond = torch.zeros_like(c).to(x)
            
        x_uncond = copy.deepcopy(x)
        if latent_uncond is not None:
            x_uncond[:,-latent_uncond.shape[1]:] = latent_uncond


        uncond_pred = self.model(
            x_uncond, 
            t, 
            encoder_hidden_states=c_uncond,
            global_hidden_states=None,              # Removed in this version.
            rotary_embedding=rotary_embedding,
        ).sample
        if w < 1.0:     # 0.0 for unconditional
            return uncond_pred
            
        cond_pred = self.model(
            x, 
            t, 
            encoder_hidden_states=c,
            global_hidden_states=None,              # Removed in this version.
            rotary_embedding=rotary_embedding,
        ).sample

        pred = uncond_pred + w * (cond_pred - uncond_pred)

        if latent_uncond is not None:
            pred[:,-latent_uncond.shape[1]:] = 0

        return pred


class VAFlow(pl.LightningModule):
    def __init__(self, 
                 # Model setting
                 ckpt_dir_image_encoder     : str = "clip",
                 ckpt_dir_audio_dit         : str = None,
                 ckpt_dir_vocoder           : str = None,
                 vaflow_ckpt_path           : str = None,
                 vaflow_custom              : bool = False,
                 vaflow_custom_config       : Dict = None,
                 resume_training            : bool = False,
                 ignore_keys                : list = list(),
                 phone_ebd_dim              : int = 32,
                 cond_feat_dim              : int = 768,
                 lip_feat_dim               : int = 1024,
                 synch_feat_dim             : int = 768,
                 dit_num_layers             : int = 24,
                 # VAE setting
                 ckpt_dir_vae               : str = None,
                 vae_latent_scaling_factor  : float = 1.0,
                 original_channel         : int = 64,
                 audio_duration_sec       : int = 10,
                 audio_length_per_sec     : int = 25,
                 # Training setting
                 lr_warmup_steps        : int = 2000,
                 use_cache_video_feat   : bool = False,
                 scale_factor           : float = 1.0,
                 unconditional_prob     : float = 0.3,
                 # Val and infer setting  
                 guidance_scale         : float = 3.0,
                 sample_steps           : int = 50,
                 sample_method          : str = "dopri5",
                 audio_sample_rate      : int = 44100,
                 num_samples_per_prompt : int = 1,
                 # PL training setting 
                 monitor        : str = None, 
                 log_data_time  : bool = True,
                 ):
        super().__init__()
        
        ''' Model building '''
        self.image_encoder = CLIPViT(ckpt_dir_image_encoder)
        self.vae = AutoencoderKL.from_pretrained(
            # From pretrained
            ckpt_dir_vae,
            local_files_only=True,
            scaling_factor=vae_latent_scaling_factor,
            low_cpu_mem_usage=False, 
            ignore_mismatched_sizes=False,
            use_safetensors=True,
        )
        self.vocoder = SpeechT5HifiGan.from_pretrained(
            # From pretrained
            ckpt_dir_vocoder,
            local_files_only=True,
            low_cpu_mem_usage=True, 
            ignore_mismatched_sizes=False,
            use_safetensors=True,
        )

        if vaflow_custom:
            self.vaflow = instantiate_from_config(vaflow_custom_config)
            _dit_cross_attn_dim = vaflow_custom_config.params.cross_attention_dim
            self.rotary_embed_dim = vaflow_custom_config.params.attention_head_dim // 2
            self.latent_in_dim = vaflow_custom_config.params.in_channels
        else:
            self.vaflow = StableAudioDiTModel.from_pretrained(
                ckpt_dir_audio_dit, local_files_only=True,                  # From pretrained
                low_cpu_mem_usage=False, ignore_mismatched_sizes=True,      # Setting for model structure changes
                num_layers=dit_num_layers,                                  # Number of layers
                use_safetensors=True,                                       # Safe tensor
            )
            _dit_cross_attn_dim = self.vaflow.config.cross_attention_dim
            self.rotary_embed_dim = self.vaflow.config.attention_head_dim // 2
            self.latent_in_dim = self.vaflow.config.in_channels

        self.vaflow_ckpt_path = vaflow_ckpt_path
        self.resume_training = resume_training


        self.lip_proj = nn.Sequential(
            nn.Linear(1024, lip_feat_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(lip_feat_dim * 2, lip_feat_dim, bias=False),
        )
        self.synch_proj = nn.Sequential(
            nn.Linear(768, synch_feat_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(synch_feat_dim * 2, synch_feat_dim, bias=False),
        )

        self.ref_proj = nn.Sequential(
            nn.Linear(256, cond_feat_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(cond_feat_dim * 2, cond_feat_dim, bias=False),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(768, cond_feat_dim * 2, bias=False),
            nn.SiLU(),
            nn.Linear(cond_feat_dim * 2, cond_feat_dim, bias=False),
        )
        self.phone_embedding = torch.nn.Embedding(
            num_embeddings=200, 
            embedding_dim=phone_ebd_dim
        )
        self.phone_embedding.requires_grad_(True)


        
        self.latent_length = int(audio_duration_sec * audio_length_per_sec)
        self.original_channel = original_channel


        ''' Init. Load ckpt. Tuning and Lora setting. '''
        # self.init_weight()
        if vaflow_ckpt_path is not None:
            self.init_from_ckpt(vaflow_ckpt_path, ignore_keys=ignore_keys)

        for _module in [self.image_encoder, self.vae, self.vocoder]:
            _module.requires_grad_(False)

            
        ''' Training settting '''
        self.lr_warmup_steps = lr_warmup_steps
        self.use_cache_video_feat = use_cache_video_feat
        self.scale_factor = scale_factor
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.unconditional_prob = unconditional_prob
        assert 0.0 <= self.unconditional_prob <= 1.0, "Unconditional_prob should be in [0.0, 1.0]"
        # self.video_interpolate_mode = video_interpolate_mode

        # Val and infer
        self.guidance_scale = guidance_scale
        self.sample_steps = sample_steps
        self.sample_method = sample_method
        self.audio_sample_rate = audio_sample_rate
        self.num_samples_per_prompt = num_samples_per_prompt
        
        # PL training setting
        self.log_data_time = log_data_time
        if self.log_data_time:
            self.last_log_data_time = time.time()
        if monitor is not None:
            self.monitor = monitor

    # ### 
    #     self.apply(self._init_weights)
    
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_uniform_(module.weight)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Conv2d):
    #         nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
    #     elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
    #         nn.init.constant_(module.weight, 1)
    #         nn.init.constant_(module.bias, 0)


    def init_from_ckpt(self, ckpt_path, ignore_keys=list()):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("=> Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if len(ignore_keys) > 0:
            self.load_state_dict(sd, strict=False)
        else:
            self.load_state_dict(sd, strict=True)
        print(f"=> Restored vaflow ckpt from {ckpt_path}")
    

    def init_weight(self):
        pass

    def configure_optimizers(self):
        # LR.
        lr = self.learning_rate

        # Optimizer.
        optimizer = torch.optim.AdamW( 
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=lr, 
        )

            
        # Scheduler.
        def fn(warmup_steps, step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            else:
                return 1.0
        def linear_warmup_decay(warmup_steps):
            return partial(fn, warmup_steps)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(self.lr_warmup_steps),
            ),
            "interval": "step",
            "frequency": 1,
        }


        if hasattr(self, 'trainer') and self.trainer is not None:
            print(f"Current Epoch {self.trainer.current_epoch}, Current Steps {self.global_step}, Current Log Dir {self.trainer.logger.save_dir}")
        else:
            print("Trainer not initialized yet, epoch and step will be restored from checkpoint")

        return [optimizer], [scheduler]



    def get_input(self, batch, use_cache_video_feat=False):
        video_data_key = "video_frame" if not use_cache_video_feat else "video_feat"

        # Base input
        video_data, video_id, video_path = batch[video_data_key], batch["video_id"], batch["video_path"]
        ref_audio_ebd, audio_waveform, audio_sr, audio_duration = batch["ref_audio_ebd"], batch["audio_waveform"], batch["audio_sr"], batch["audio_duration"]
        synch_feat, lip_feat, phone_id, phone_seq = batch['synch_feat'], batch['lip_feat'], batch['phone_id'], batch['phone_seq']
        
        # Type convert
        video_data = video_data.float()        
        audio_waveform = audio_waveform.float()
        
        return video_data, video_id, video_path, \
                ref_audio_ebd, audio_waveform, audio_sr, audio_duration,\
                synch_feat, lip_feat, \
                phone_id, phone_seq
    
    
    """ Fit (train) setting """
    def on_fit_start(self):
        pass
        
    """ Training """
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=0):
        if self.log_data_time:
            data_time = time.time()-self.last_log_data_time
            data_time = torch.tensor(data_time)
            self.log("custom/data_time", data_time, batch_size=len(batch), sync_dist=True, 
                     on_step=True, on_epoch=True, prog_bar=True, logger=True,)
               
    def training_step(self, batch, batch_idx, dataloader_idx=0, custom_device=None):
        #[bs, 100, 768]  list(bs)     [bs, 256]               list(bs)  list(bs)                       [bs, 250, 1024]   [bs, 250]   
        video_feat,     video_id, _, ref_audio_ebd, waveform, audio_sr, audio_duration,  synch_feature, lip_feature,      phone_id,   _ = self.get_input(batch, self.use_cache_video_feat)
        batch_size = video_feat.shape[0]
        device = video_feat.device

        # [bs, sr*10 + ..]   [bs, mel_length(sr*duration/hop_length), n_mel]
        waveform,            log_mel_spec = extract_batch_mel(waveform.squeeze(1), cut_audio_duration = 10, sampling_rate = self.audio_sample_rate, hop_length = 160, maximum_amplitude = 0.5,
                                                             filter_length = 1024, n_mel = 64, mel_fmin = 0, mel_fmax = 8000, win_length = 1024)
        log_mel_spec = log_mel_spec.unsqueeze(1)  # [bs, 1, target_mel_length (sr*duration/hop_length, 1000), n_mel (64)]
        
        
        """ 1. Pad features """
        video_feat  = torch.nn.functional.interpolate(video_feat.permute(0, 2, 1), size=self.latent_length, mode='nearest')                       # [bs, 768, latent_length]
        video_feat  = video_feat.transpose(1, 2)                                                                                                  # [bs, latent_length, 768]
        lip_feature = torch.nn.functional.pad(lip_feature, (0, 0, 0, self.latent_length - lip_feature.shape[1], 0, 0), mode='constant', value=0)  # [bs, latent_length, 1024] 
        phone_id    = torch.nn.functional.pad(phone_id,    (0,       self.latent_length - phone_id.shape[1]),          mode='constant', value=0)  # [bs, latent_length]   
        synch_feature = synch_feature.reshape([batch_size, -1, synch_feature.shape[-1]])                                                          # [bs, length_synch (10 * 3 * 8), 768]   
        synch_feature = synch_feature.transpose(1, 2)                                                                                             # [bs, 768, length_synch]   
        synch_feature = torch.nn.functional.interpolate(synch_feature, size=self.latent_length, mode='nearest-exact')                             # [bs, 768, latent_length]   
        synch_feature = synch_feature.transpose(1, 2)                                                                                             # [36, latent_length, 768]


        """ 2. Proj features """
        video_feat_cond = self.cond_proj(video_feat)                                # [bs, latent_length, 768] 
        ref_speech_cond = self.ref_proj(ref_audio_ebd.unsqueeze(1))                 # [bs, 1, 768]
        phone_latent = self.phone_embedding(phone_id)                               # [bs, latent_length, phone_latent_dim]
        synch_feature = self.synch_proj(synch_feature)                              # [bs, latent_length, synch_feat_dim]
        lip_feature = self.lip_proj(lip_feature)                                    # [bs, latent_length, lip_feat_dim]
    
        # crossattn cond
        video_feat_cond     = torch.concat([ref_speech_cond, video_feat_cond], dim = 1)                      # [bs, 1+latent_length, 768]
        # ref_speech_cond_pad = torch.zeros([batch_size, 1, phone_latent.shape[-1]]).to(device)                # [bs, 1, phone_latent_dim]
        # phone_latent        = torch.concat([ref_speech_cond_pad, phone_latent], dim = 1)                     # [bs, 1+latent_length, phone_latent_dim]
        # cross_attn_cond     = torch.concat([video_feat_cond,  phone_latent], dim = -1)    # [bs, 1+latent_length, 768 + phone_latent_dim]
        cross_attn_cond     = video_feat_cond                                             # [bs, 1+latent_length, 768]

        # concat cond
        phone_latent  = phone_latent.transpose(1,2)                                      # [bs, phone_latent_dim, latent_length]
        synch_feature = synch_feature.transpose(1,2)                                     # [bs, synch_feat_dim, latent_length]
        lip_feature = lip_feature.transpose(1,2)                                         # [bs, lip_feat_dim, latent_length]
        latent_cond = torch.concat([lip_feature, phone_latent, synch_feature], dim=1)    # [bs, synch_feat_dim + phone_latent_dim + lip_feat_dim (1824), latent_length]



        """ 3. Encode audio waveform to latents. """
        with torch.no_grad():
            audio_latent = self.vae.encode(log_mel_spec.to(self.vae.encoder.conv_in.weight.dtype)).latent_dist    #  [bs, 8, target_mel_length/4(250), n_mel/4(16)]
            audio_latent = audio_latent.sample()                                                                  #  [bs, 8, target_mel_length/4(250), n_mel/4(16)]
            audio_latent = audio_latent.transpose(-2,-1).transpose(-2,-3)                                         #  [bs, n_mel/4(16), 8, target_mel_length/4(250)]
            audio_latent = audio_latent.reshape([batch_size, -1, audio_latent.shape[-1]])                         #  [bs, original_channel(8*16), target_mel_length/4(250)]
        audio_latent = audio_latent.detach()                         #  [bs, original_channel, target_mel_length/4(250)]
        audio_latent = audio_latent * self.scale_factor
        video_latent = torch.randn_like(audio_latent).detach()       #  [bs, original_channel, target_mel_length/4(250)]



        # TODO: Pad Transcript information to audio_latent
        if random.random() < self.unconditional_prob:
            latent_cond = torch.zeros_like(latent_cond, dtype = latent_cond.dtype).to(device)    # [bs, latent_length, padded_phone_length]
            cross_attn_cond = torch.zeros_like(cross_attn_cond, dtype=cross_attn_cond.dtype).to(device)                           # [bs, padded_phone_length]



        """ 4. Sample time step for DIT. """
        video_latent = torch.cat([video_latent, latent_cond], dim=1)                   # [bs, original_channel + synch_feat_dim + lip_feat_dim, latent_length]
        audio_latent = torch.cat([audio_latent, latent_cond], dim=1)                   # [bs, original_channel + synch_feat_dim + lip_feat_dim, latent_length]
        t = torch.rand(video_latent.shape[0]).to(device)    # [bs]
        # # # ### CHECK LATENT
        # # # with torch.no_grad():
        # # #     audio_latent_ = audio_latent[:, :self.original_channel, :]  # [bs, original_channel, latent_length]
        # # #     audio_latent_ = audio_latent_ / self.scale_factor
        # # #     audio_latent_ = audio_latent_.reshape([batch_size, -1, 8, audio_latent_.shape[-1]])   # [bs, 16, 8, latent_length]
        # # #     audio_latent_ = audio_latent_.transpose(-2,-3).transpose(-2, -1)                     # [bs, 8, latent_length, 16]
        # # #     mel_spectrogram = self.vae.decode(audio_latent_).sample                             # [bs, 1, target_mel_length(latent_length*4), 64(16*4)]
        # # #     gen_audio = self.vocoder(mel_spectrogram.squeeze(1))                               # [bs, duration*sr+...]
        # # #     gen_audio = gen_audio.cpu()
        # # #     for i, audio in enumerate(gen_audio):
        # # #         audio = audio[:int(audio_sr[i]*audio_duration[i])].unsqueeze(0)                # [1, duration*sr]
        # # #         audio_path = os.path.join('log', "{}.wav".format(video_id[i]))
        # # #         torchaudio.save(audio_path, audio, self.audio_sample_rate) 


        """ 5. Sample probability path. """
        path_sample = self.path.sample(t=t, x_0=video_latent, x_1=audio_latent)
        dx_t = path_sample.dx_t  # [bs, original_channel + synch_feat_dim + lip_feat_dim, latent_length]
        x_t = path_sample.x_t    # [bs, original_channel + synch_feat_dim + lip_feat_dim, latent_length]
        t = path_sample.t        # [bs]


        """ 6. Forward VAFlow. """
        # tuple([x_t.shape[-1] + 1, self.rotary_embed_dim], [x_t.shape[-1] + 1, self.rotary_embed_dim])
        rotary_embedding = get_1d_rotary_pos_embed(
            self.rotary_embed_dim,
            x_t.shape[2] + 1,      
            use_real=True,
            repeat_interleave_real=False,
        )
        
        # [bs, original_channel + synch_feat_dim + lip_feat_dim, latent_length]
        audio_latent_pred = self.vaflow(
            x_t, 
            t, 
            encoder_hidden_states=cross_attn_cond,
            global_hidden_states=None,              
            rotary_embedding=rotary_embedding,
        ).sample
        

        """ 7. Compute loss. """
        dx_t = dx_t[:, :self.original_channel, :]                            # [bs, original_channel, latent_length]
        audio_latent_pred = audio_latent_pred[:, :self.original_channel, :]  # [bs, original_channel, latent_length]
        loss = nn_func.mse_loss(audio_latent_pred, dx_t, reduction="mean")

        has_nan_loss = torch.isnan(loss).any().item()
        assert not has_nan_loss, "Loss has nan value!"


        ''' 8. Log '''
        self.log("train/loss", loss, batch_size=batch_size, sync_dist=True, 
                    on_step=True, on_epoch=True, prog_bar=True, logger=True,)


        return loss
        
    def on_train_batch_end(self, batch, batch_idx, dataloader_idx=0):
        if self.log_data_time:
            self.last_log_data_time = time.time()

    """ Validation """
    def on_validation_epoch_start(self):
        # Set dirs
        self.val_log_dir = os.path.join(self.trainer.logger.save_dir, "val")
        os.makedirs(self.val_log_dir, exist_ok=True)
        self.val_log_dir_for_audio_per_epoch = os.path.join(self.val_log_dir, "audio", "epoch_{:04d}_global_step_{:.2e}".format(self.trainer.current_epoch, self.global_step))
        self.val_log_dir_for_speech_per_epoch = os.path.join(self.val_log_dir, "speech", "epoch_{:04d}_global_step_{:.2e}".format(self.trainer.current_epoch, self.global_step))
        os.makedirs(self.val_log_dir_for_audio_per_epoch, exist_ok=True)
        os.makedirs(self.val_log_dir_for_speech_per_epoch, exist_ok=True)
        
    def validation_step_(self, batch, batch_idx, dataloader_idx=0, custom_device=None):
        #[bs, 100, 768]  list(bs)     [bs, 256]          list(bs)  list(bs)                       [bs, length_l, 1024]   [bs, length_p]   
        video_feat,     video_id, _, ref_audio_ebd, _, audio_sr, audio_duration,  synch_feature, lip_feature,            phone_id,   _ = self.get_input(batch, self.use_cache_video_feat)
        batch_size = video_feat.shape[0]
        device = video_feat.device

        wrapped_vaflow = WrappedModel(self.vaflow)
        solver = ODESolver(velocity_model=wrapped_vaflow) 

        # # # # [bs, sr*10 + ..]   [bs, mel_length(sr*duration/hop_length), n_mel]
        # # # waveform,            log_mel_spec = extract_batch_mel(waveform.squeeze(1), cut_audio_duration = 10, sampling_rate = self.audio_sample_rate, hop_length = 160, maximum_amplitude = 0.5,
        # # #                                                      filter_length = 1024, n_mel = 64, mel_fmin = 0, mel_fmax = 8000, win_length = 1024)
        # # # log_mel_spec = log_mel_spec.unsqueeze(1)  # [bs, 1, target_mel_length (sr*duration/hop_length, 1000), n_mel (64)]
        
        
        """ 1. Pad features """
        video_feat  = torch.nn.functional.interpolate(video_feat.permute(0, 2, 1), size=self.latent_length, mode='nearest')                       # [bs, 768, latent_length]
        video_feat  = video_feat.transpose(1, 2)                                                                                                  # [bs, latent_length, 768]
        lip_feature = torch.nn.functional.pad(lip_feature, (0, 0, 0, self.latent_length - lip_feature.shape[1], 0, 0), mode='constant', value=0)  # [bs, latent_length, 1024] 
        phone_id    = torch.nn.functional.pad(phone_id,    (0,       self.latent_length - phone_id.shape[1]),          mode='constant', value=0)  # [bs, latent_length]   
        synch_feature = synch_feature.reshape([batch_size, -1, synch_feature.shape[-1]])                                                          # [bs, length_synch (10 * 3 * 8), 768]   
        synch_feature = synch_feature.transpose(1, 2)                                                                                             # [bs, 768, length_synch]   
        synch_feature = torch.nn.functional.interpolate(synch_feature, size=self.latent_length, mode='nearest-exact')                             # [bs, 768, latent_length]   
        synch_feature = synch_feature.transpose(1, 2)                                                                                             # [36, latent_length, 768]


        """ 2. Proj features """
        video_feat_cond = self.cond_proj(video_feat)                                # [bs, latent_length, 768] 
        ref_speech_cond = self.ref_proj(ref_audio_ebd.unsqueeze(1))                 # [bs, 1, 768]
        phone_latent = self.phone_embedding(phone_id)                               # [bs, latent_length, phone_latent_dim]
        synch_feature = self.synch_proj(synch_feature)                              # [bs, latent_length, synch_feat_dim]
        lip_feature = self.lip_proj(lip_feature)                                    # [bs, latent_length, lip_feat_dim]
        
        # # # crossattn cond
        # # video_feat_cond     = torch.concat([ref_speech_cond, video_feat_cond], dim = 1)                      # [bs, 1+latent_length, 768]
        # # ref_speech_cond_pad = torch.zeros([batch_size, 1, phone_latent.shape[-1]]).to(device)                # [bs, 1, phone_latent_dim]
        # # phone_latent        = torch.concat([ref_speech_cond_pad, phone_latent], dim = 1)                     # [bs, 1+latent_length, phone_latent_dim]
        # # cross_attn_cond     = torch.concat([video_feat_cond,  phone_latent], dim = -1)    # [bs, 1+latent_length, 768 + phone_latent_dim]
        # # cross_attn_cond_uncond = torch.zeros_like(cross_attn_cond)              # [bs, 1+latent_length, 768 + phone_latent_dim]
        
        # # # concat cond
        # # synch_feature = synch_feature.transpose(1,2)                                # [bs, synch_feat_dim, latent_length]
        # # lip_feature = lip_feature.transpose(1,2)                                    # [bs, lip_feat_dim, latent_length]
        # # latent_uncond = None
        # # latent_cond = torch.concat([lip_feature, synch_feature], dim=1)                   # [bs, synch_feat_dim + lip_feat_dim (1792), latent_length]
        # # latent_uncond = torch.zeros_like(latent_cond)                                     # [bs, synch_feat_dim + lip_feat_dim (1792)， latent_length]  

        # crossattn cond
        video_feat_cond     = torch.concat([ref_speech_cond, video_feat_cond], dim = 1)                      # [bs, 1+latent_length, 768]
        # ref_speech_cond_pad = torch.zeros([batch_size, 1, phone_latent.shape[-1]]).to(device)                # [bs, 1, phone_latent_dim]
        # phone_latent        = torch.concat([ref_speech_cond_pad, phone_latent], dim = 1)                     # [bs, 1+latent_length, phone_latent_dim]
        # cross_attn_cond     = torch.concat([video_feat_cond,  phone_latent], dim = -1)    # [bs, 1+latent_length, 768 + phone_latent_dim]
        cross_attn_cond     = video_feat_cond                                             # [bs, 1+latent_length, 768]
        cross_attn_cond_uncond = torch.zeros_like(cross_attn_cond)                        # [bs, 1+latent_length, 768]

        # concat cond
        phone_latent  = phone_latent.transpose(1,2)                                      # [bs, phone_latent_dim, latent_length]
        synch_feature = synch_feature.transpose(1,2)                                     # [bs, synch_feat_dim, latent_length]
        lip_feature = lip_feature.transpose(1,2)                                         # [bs, lip_feat_dim, latent_length]
        latent_uncond = None
        latent_cond = torch.concat([lip_feature, phone_latent, synch_feature], dim=1)    # [bs, synch_feat_dim + phone_latent_dim + lip_feat_dim (1824), latent_length]
        latent_uncond = torch.zeros_like(latent_cond)                                    # [bs, synch_feat_dim + phone_latent_dim + lip_feat_dim (1824)， latent_length]  



        _seeds = torch.randint(0, 10000, size=(self.num_samples_per_prompt,), device=device)
        for _rand_i in range(self.num_samples_per_prompt):
            generator = torch.Generator(device=device).manual_seed(_seeds[_rand_i].item())
            video_latent = randn_tensor((batch_size, self.original_channel, self.latent_length), device=device, generator=generator) # [bs, original_channel, latent_length]
            video_latent = torch.cat([video_latent, latent_cond], dim=1)                   # [bs, original_channel + synch_feat_dim + lip_feat_dim, latent_length]
            

            # 1.2 Rotary embedding.
            rotary_embedding = get_1d_rotary_pos_embed(
                self.rotary_embed_dim,
                video_latent.shape[2] + 1,
                use_real=True,
                repeat_interleave_real=False,
            )

            # 2. Flow matching.
            step_size = None
            time_grid = torch.linspace(0, 1, self.sample_steps+1, ).to(device) 
            synthetic_samples = solver.sample(
                time_grid=time_grid,
                x_init=video_latent,
                method=self.sample_method,
                return_intermediates=False,
                atol=1e-5,
                rtol=1e-5,
                step_size=step_size,
                latent_uncond=latent_uncond,
                c=cross_attn_cond,
                cross_attn_uncond=cross_attn_cond_uncond,
                w=self.guidance_scale,
                rotary_embedding=rotary_embedding,
            )

            # 3. Log to audio.
            audio_latent = synthetic_samples                           # [bs, original_channel + synch_feat_dim + lip_feat_dim, latent_length]
            audio_latent = audio_latent[:, :self.original_channel, :]  # [bs, original_channel, latent_length]
            audio_latent = audio_latent / self.scale_factor

            audio_latent = audio_latent.reshape([batch_size, -1, 8, audio_latent.shape[-1]])   # [bs, 16, 8, latent_length]
            audio_latent = audio_latent.transpose(-2,-3).transpose(-2, -1)                     # [bs, 8, latent_length, 16]
            
            mel_spectrogram = self.vae.decode(audio_latent).sample                             # [bs, 1, target_mel_length(latent_length*4), 64(16*4)]
            gen_audio = self.vocoder(mel_spectrogram.squeeze(1))                               # [bs, duration*sr+...]
            gen_audio = gen_audio.cpu()

            for i, audio in enumerate(gen_audio):
                audio = audio[:int(audio_sr[i]*audio_duration[i])].unsqueeze(0)                # [1, duration*sr]
                as_path = self.val_log_dir_for_audio_per_epoch if phone_id[i].sum() == 0 else self.val_log_dir_for_speech_per_epoch
                audio_path = os.path.join(as_path, "{}.wav".format(video_id[i]))
                if self.num_samples_per_prompt != 1:
                    audio_path = os.path.join(as_path, "{}_{:02d}.wav".format(video_id[i], _rand_i))

                torchaudio.save(audio_path, audio, self.audio_sample_rate) 
            
        return 0.0

    def validation_step(self, batch, batch_idx, dataloader_idx=0, custom_device=None):
        with torch.no_grad():
            self.predict = False
            self.validation_step_(batch=batch, batch_idx=batch_idx)
        return 0.0


    def validation_epoch_end(self, val_step_outputs):
        self.log("val/loss_temp", 0.0, sync_dist=True, prog_bar=True, logger=True,)
        
    """ Prediction """
    def on_predict_epoch_start(self):
        # Set dirs
        self.val_log_dir = os.path.join(self.trainer.logger.save_dir, "predict")
        os.makedirs(self.val_log_dir, exist_ok=True)
        self.val_log_dir_for_audio_per_epoch = os.path.join(self.val_log_dir, "audio")
        self.val_log_dir_for_speech_per_epoch = os.path.join(self.val_log_dir, "speech")
        os.makedirs(self.val_log_dir_for_audio_per_epoch, exist_ok=True)
        os.makedirs(self.val_log_dir_for_speech_per_epoch, exist_ok=True)


    def predict_step(self, batch, batch_idx, dataloader_idx=0,):
        with torch.no_grad():
            self.predict = True
            self.validation_step_(batch=batch, batch_idx=batch_idx)
            self.predict = False
        return 0.0

    @torch.no_grad()
    def encode_image(self, image, use_projection=False):
        # [B, F, C, H, W] -> [B F, C, H, W]
        _b = image.shape[0]
        x = einops.rearrange(image, "b f c h w -> (b f) c h w")
        x = self.image_encoder(x)

        x = self.image_encoder.visual.ln_post(x[:, 0, :])   # Get CLS token only
        if use_projection:
            if self.image_encoder.visual.proj is not None:
                x = x @ self.image_encoder.visual.proj
        x = einops.rearrange(x, "(b f) c -> b f c", b=_b)
        
        return x
    