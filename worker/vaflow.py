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

# from peft import LoraConfig, get_peft_model
# from peft.utils import get_peft_model_state_dict

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

from agc import AGC

from .base import instantiate_from_config, get_class_from_config
from model.clip.clip_module import CLIPViT


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        t = (t * 1000).long()
        return self.model(x, t)


class VAFlow(pl.LightningModule):
    def __init__(self, 
                 # Model setting
                 ckpt_dir_image_encoder     : str = "clip",
                 videoclipvae_config        : Dict = None,
                 videoclipvae_ckpt_path     : str = None,
                 audiocodec_ckpt_path       : str = None,
                 dit_config                 : Dict = None,
                 dit_ckpt_path              : str = None,
                 vaflow_ckpt_path           : str = None,
                 ignore_keys                : list = list(),
                 # Training setting
                 lr_warmup_steps        : int = 2000,
                 use_cache_video_feat   : bool = False,
                 scale_factor           : float = 1.0,
                 # Val and infer setting  
                 # PL training setting 
                 monitor        : str = None, 
                 log_data_time  : bool = True,
                 ):
        super().__init__()
        
        ''' Model building '''
        self.image_encoder = CLIPViT(ckpt_dir_image_encoder)
        self.video_vae = instantiate_from_config(videoclipvae_config)
        self.audio_codec = AGC.from_pretrained(audiocodec_ckpt_path)
        self.vaflow = instantiate_from_config(dit_config)
        
        ''' Init. Load ckpt. Tuning and Lora setting. '''
        self.init_weight()
        if videoclipvae_ckpt_path is not None:
            self.init_submodules_from_ckpt(videoclipvae_ckpt_path, "video_vae", )
        if vaflow_ckpt_path is not None:
            self.init_from_ckpt(vaflow_ckpt_path, ignore_keys=ignore_keys)
            
        # Freeze.
        for _module in [self.image_encoder, self.video_vae, self.audio_codec]:
            _module.requires_grad_(False)
            
        ''' Training settting '''
        self.lr_warmup_steps = lr_warmup_steps
        self.use_cache_video_feat = use_cache_video_feat
        self.scale_factor = scale_factor
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # Val and infer
        
        # PL training setting
        self.log_data_time = log_data_time
        if self.log_data_time:
            self.last_log_data_time = time.time()
        if monitor is not None:
            self.monitor = monitor
        
    def init_from_ckpt(self, ckpt_path, ignore_keys=list()):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("=> Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=True)
        print(f"=> Restored vae ckpt from {ckpt_path}")
    
    def init_submodules_from_ckpt(self, ckpt_path, module_name, ignore_keys=list()):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt["state_dict"] # NOTE Assume that the key is "state_dict" in pytorch-lightning ckpt.
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("=> Deleting key {} from state_dict.".format(k))
                    del sd[k]
        if module_name == "video_vae":
            video_vae_sd = {}
            for k, v in sd.items():
                if k.startswith("video_vae."):
                    video_vae_sd[k.replace("video_vae.", "")] = v
            self.video_vae.load_state_dict(video_vae_sd, strict=True)
            print(f"=> Restored {module_name} ckpt from {ckpt_path}")
        else:
            raise ValueError(f"Module name {module_name} is not in self.__dict__ ")
    
    def init_weight(self):
        # TODO Better weight initialization for VAE
        pass

    def configure_optimizers(self):
        # LR.
        lr = self.learning_rate

        # Optimizer.
        # TODO Any other better setting
        optimizer = torch.optim.AdamW( 
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=lr, 
        )

        # Scheduler.
        # TODO Any other better scheduler
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

        return [optimizer], [scheduler]

    # def on_save_checkpoint(self, checkpoint):
    #     if self.use_ema_dit:
    #         checkpoint['state_dict_ema_dit'] = self.ema_dit.state_dict()
            
    # def on_load_checkpoint(self, checkpoint):
        # if self.use_ema_dit:
        #     self.ema_dit.load_state_dict(checkpoint['state_dict_ema_dit'])

    def get_input(self, batch, use_cache_video_feat=False):
        video_data_key = "video_frame" if not use_cache_video_feat else "video_feat"
        # Base input
        video_data, video_id, video_path, audio_waveform, audio_sr = batch[video_data_key], batch["video_id"], batch["video_path"], batch["audio_waveform"], batch["audio_sr"]
        # Type convert
        video_data = video_data.float()        
        audio_waveform = audio_waveform.float()
        
        return video_data, video_id, video_path, audio_waveform, audio_sr
    
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
        video_frame, video_id, _, waveform, *_ = self.get_input(batch, self.use_cache_video_feat)
        batch_size = video_frame.shape[0]
        device = video_frame.device
        
        """ 1. Encode video frame to vae latents. """
        # Encode video frame to CLIP embedding.
        if self.use_cache_video_feat:
            video_feat = video_frame                        # [B, F, C]
        else:
            video_feat = self.encode_image(video_frame)     # [B, F, C]
            video_feat = video_feat.detach()
        # Forward VAE.
        video_feat = einops.rearrange(video_feat, "b f c -> b c f")
        with torch.no_grad():
            video_latent = self.video_vae.encode(video_feat)
        # video_latent = video_latent.mode().detach()
        video_latent = video_latent.sample().detach()       # [B, C, F]
        video_latent = einops.rearrange(video_latent, "b c f -> b f c")
        
        """ 2. Encode audio waveform to latents. """
        with torch.no_grad():
            audio_latent = self.audio_codec.encode(waveform)# [B, 32, 1000]
        audio_latent = einops.rearrange(audio_latent, "b c (f1 f2) -> b f2 (c f1)", f1=10)
        audio_latent = audio_latent.detach()                # [B, 100, 320,]
        audio_latent = audio_latent * self.scale_factor
        
        """ 3. Sample time step for DIT. """
        t = torch.rand(video_latent.shape[0]).to(device) 
        
        """ 4. Sample probability path. """
        path_sample = self.path.sample(t=t, x_0=video_latent, x_1=audio_latent)
        dx_t = path_sample.dx_t
        x_t = path_sample.x_t
        t = path_sample.t
        t = (t * 1000).long()
        
        """ 5. Forward VAFlow. """
        audio_latent_pred = self.vaflow(x=x_t, t=t)
        
        """ 6. Compute loss. """
        loss = nn_func.mse_loss(audio_latent_pred, dx_t, reduction="mean")

        # Loss nan check
        has_nan_loss = torch.isnan(loss).any().item()
        assert not has_nan_loss, "Loss has nan value!"
        
        ''' 7. Log '''
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
        self.val_log_dir_for_video_per_epoch = os.path.join(self.val_log_dir, "video", "epoch_{:04d}_global_step_{:.2e}".format(self.trainer.current_epoch, self.global_step))
        os.makedirs(self.val_log_dir_for_video_per_epoch, exist_ok=True)
        
    def validation_step(self, batch, batch_idx, dataloader_idx=0, custom_device=None):
        video_frame, video_id, *_ = self.get_input(batch, self.use_cache_video_feat)
        batch_size = video_frame.shape[0]
        device = video_frame.device

        wrapped_vaflow = WrappedModel(self.vaflow)
        solver = ODESolver(velocity_model=wrapped_vaflow) 
        
        # 1. Encode video frame to latents.
        if self.use_cache_video_feat:
            video_feat = video_frame                        # [B, F, C]
        else:
            video_feat = self.encode_image(video_frame)     # [B, F, C]
            video_feat = video_feat.detach()
        video_feat = einops.rearrange(video_feat, "b f c -> b c f")
        with torch.no_grad():
            video_latent = self.video_vae.encode(video_feat)
        # video_latent = video_latent.mode().detach()
        video_latent = video_latent.sample().detach()       # [B, C, F]
        video_latent = einops.rearrange(video_latent, "b c f -> b f c")

        # 2. Flow matching.
        step_size = None
        # time_grid = torch.tensor([0.0, 1.0], device=device)
        time_grid = torch.linspace(0, 1, 50, ).to(device) 
        synthetic_samples = solver.sample(
            time_grid=time_grid,
            x_init=video_latent,
            method='midpoint',
            return_intermediates=False,
            atol=1e-5,
            rtol=1e-5,
            step_size=step_size,
        )

        # 3. Log to audio.
        audio_latent = synthetic_samples
        audio_latent = einops.rearrange(audio_latent, "b f2 (c f1)-> b c (f1 f2)", f1=10)
        audio_latent = audio_latent / self.scale_factor
        gen_audio = self.audio_codec.decode(audio_latent)
        gen_audio = gen_audio.cpu()
        for i, audio in enumerate(gen_audio):
            audio_path = os.path.join(self.val_log_dir_for_video_per_epoch, f"{video_id[i]}.wav")
            torchaudio.save(audio_path, audio, 48000)        
            
        return 0.0
    
    def validation_epoch_end(self, val_step_outputs):
        
        self.log("val/loss_temp", 0.0, sync_dist=True, prog_bar=True, logger=True,)
        
    """ Prediction """
    def on_predict_epoch_start(self):
        # Set dirs
        self.val_log_dir = os.path.join(self.trainer.logger.save_dir, "predict")
        os.makedirs(self.val_log_dir, exist_ok=True)
        self.val_log_dir_for_video_per_epoch = os.path.join(self.val_log_dir, "video")
        os.makedirs(self.val_log_dir_for_video_per_epoch, exist_ok=True)
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0,):
        return self.validation_step(batch=batch, batch_idx=batch_idx)

    @torch.no_grad()
    def encode_image(self, image, use_projection=False):
        # [B, F, C, H, W] -> [B F, C, H, W]
        _b = image.shape[0]
        x = einops.rearrange(image, "b f c h w -> (b f) c h w")
        x = self.image_encoder(x)

        # NOTE Temporally code.
        # x = x[:, 0, :]
        # return einops.rearrange(x, "(b f) c -> b f c", b=_b)

        x = self.image_encoder.visual.ln_post(x[:, 0, :])   # Get CLS token only
        if use_projection:
            if self.image_encoder.visual.proj is not None:
                x = x @ self.image_encoder.visual.proj
        x = einops.rearrange(x, "(b f) c -> b f c", b=_b)
        
        return x
    
