import os, time, random, einops, wandb, inspect
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from functools import partial
from scipy.io.wavfile import write as scipy_wav_write
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torchvision.io as tvio
import pytorch_lightning as pl
import torch.nn.functional as nn_func
from pytorch_lightning.callbacks import BaseFinetuning

# from peft import LoraConfig, get_peft_model
# from peft.utils import get_peft_model_state_dict

from .base import instantiate_from_config, get_class_from_config
from model.clip.clip_module import CLIPViT
from model.vae.vae_vanilla import VanillaVAE_1D


class VideoCLIPVAE(pl.LightningModule):
    def __init__(self, 
                 # Model setting
                 ckpt_dir_image_encoder     : str = "clip",
                 videoclipvae_config        : Dict = None,
                 videoclipvae_ckpt_path     : str = None,
                 ignore_keys                : list = list(),
                 # Training setting
                 lr_warmup_steps        : int = 2000,
                 use_cache_video_feat   : bool = False,
                 loss_kl_weight         : float = 1e-4,
                 loss_image_contrastive_weight : float = 1.0,
                 loss_video_clip_contrastive_weight : float = 1.0,
                 loss_video_contrastive_weight : float = 1.0,
                 loss_vv_ib_contrastive_weight : float = 1.0,
                 loss_va_ib_contrastive_weight : float = 1.0,
                 infonce_temperature    : float = 0.07,
                 # Val and infer setting  
                 # PL training setting 
                 monitor        : str = None, 
                 log_data_time  : bool = True,
                 ):
        super().__init__()
        
        ''' Model building '''
        self.image_encoder = CLIPViT(ckpt_dir_image_encoder)
        self.video_vae = instantiate_from_config(videoclipvae_config)
        self.ib_proj_v = nn.Linear(videoclipvae_config.params.in_channels, 1024)
        self.ib_proj_a = nn.Linear(videoclipvae_config.params.in_channels, 1024)
        
        ''' Init. Load ckpt. Tuning and Lora setting. '''
        self.init_weight()
        if videoclipvae_ckpt_path is not None:
            self.init_from_ckpt(videoclipvae_ckpt_path, ignore_keys=ignore_keys)
            
        # Freeze.
        for _module in [self.image_encoder,]:
            _module.requires_grad_(False)
            
        # Lora.
        # for _param in self.video_diffusion.parameters():
        #     _param.requires_grad_(False)
        # unet_lora_config = LoraConfig(
        #     r=4,
        #     lora_alpha=4,
        #     init_lora_weights="gaussian",
        #     target_modules=r".*temporal_transformer_blocks.*",
        # )
        # self.video_diffusion.add_adapter(unet_lora_config, adapter_name="adapter_1")
        # self.video_diffusion = get_peft_model(self.video_diffusion, unet_lora_config).to(next(self.video_diffusion.parameters()).device)
        # self.video_diffusion.print_trainable_parameters()
        
        ''' Training settting '''
        self.lr_warmup_steps = lr_warmup_steps
        self.use_cache_video_feat = use_cache_video_feat
        self.loss_kl_weight = loss_kl_weight
        self.loss_image_contrastive_weight = loss_image_contrastive_weight
        self.loss_video_clip_contrastive_weight = loss_video_clip_contrastive_weight
        self.loss_video_contrastive_weight = loss_video_contrastive_weight
        self.loss_vv_ib_contrastive_weight = loss_vv_ib_contrastive_weight
        self.loss_va_ib_contrastive_weight = loss_va_ib_contrastive_weight
        self.infonce_temperature = infonce_temperature

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
    #     if self.use_ema_dit:
    #         self.ema_dit.load_state_dict(checkpoint['state_dict_ema_dit'])

    def get_input(self, batch, use_cache_video_feat=False):
        video_data_key = "video_frame" if not use_cache_video_feat else "video_feat"
        # Base input
        video_data, video_id, video_path = batch[video_data_key], batch["video_id"], batch["video_path"]
        ib_video_feat = batch.get("ib_video_feat", None)
        ib_audio_feat = batch.get("ib_audio_feat", None)
        # Type convert
        video_data = video_data.float()        
        if ib_video_feat is not None:
            ib_video_feat = ib_video_feat.float()
        if ib_audio_feat is not None:
            ib_audio_feat = ib_audio_feat.float()
        
        return video_data, video_id, video_path, ib_video_feat, ib_audio_feat
    
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
            
    def compute_losses(self, v_feat, v_feat_rec, ib_video, ib_audio):
        # MSE
        loss_mse = nn_func.mse_loss(v_feat_rec, v_feat, reduction="mean")
        
        # Image contrastive loss
        v_feat_proj = einops.rearrange(v_feat, "b f c -> (b f) c")
        v_feat_rec_proj = einops.rearrange(v_feat_rec, "b f c -> (b f) c")
        if self.image_encoder.visual.proj is not None:
            v_feat_proj = v_feat_proj @ self.image_encoder.visual.proj
            v_feat_rec_proj = v_feat_rec_proj @ self.image_encoder.visual.proj
        loss_image_contrastive = self.info_nce_loss(v_feat_rec_proj, v_feat_proj, temperature=self.infonce_temperature)
        
        # Video clip contrastive loss
        v_clip_feat = self.select_random_segments_with_gradients(v_feat, segment_length=20, num_segments=3)
        v_clip_feat_rec = self.select_random_segments_with_gradients(v_feat_rec, segment_length=20, num_segments=3)
        v_clip_feat = v_clip_feat.mean(dim=2)
        v_clip_feat_rec = v_clip_feat_rec.mean(dim=2)
        v_clip_feat_proj = einops.rearrange(v_clip_feat, "b s c -> (b s) c")
        v_clip_feat_rec_proj = einops.rearrange(v_clip_feat_rec, "b s c -> (b s) c")
        if self.image_encoder.visual.proj is not None:
            v_clip_feat_proj = v_clip_feat_proj @ self.image_encoder.visual.proj
            v_clip_feat_rec_proj = v_clip_feat_rec_proj @ self.image_encoder.visual.proj
        loss_video_clip_contrastive = self.info_nce_loss(v_clip_feat_rec_proj, v_clip_feat_proj, temperature=self.infonce_temperature)
        
        # Video contrastive loss
        v_full_feat = v_feat.mean(dim=1)
        v_full_feat_rec = v_feat_rec.mean(dim=1)
        if self.image_encoder.visual.proj is not None:
            v_full_feat_proj = v_full_feat @ self.image_encoder.visual.proj
            v_full_feat_rec_proj = v_full_feat_rec @ self.image_encoder.visual.proj
        loss_video_contrastive = self.info_nce_loss(v_full_feat_rec_proj, v_full_feat_proj, temperature=self.infonce_temperature)
        
        # Video IB contrastive loss
        if ib_video is not None:
            v_full_feat_rec_ib_v = self.ib_proj_v(v_full_feat_rec)
            loss_vv_ib_contrastive = self.info_nce_loss(v_full_feat_rec_ib_v, ib_video, temperature=self.infonce_temperature)
        else:
            loss_vv_ib_contrastive = .0
        if ib_audio is not None:
            v_full_feat_rec_ib_a = self.ib_proj_a(v_full_feat_rec)
            loss_va_ib_contrastive = self.info_nce_loss(v_full_feat_rec_ib_a, ib_audio, temperature=self.infonce_temperature)
        else:
            loss_va_ib_contrastive = .0
            
        return loss_mse, loss_image_contrastive, loss_video_clip_contrastive, loss_video_contrastive, loss_vv_ib_contrastive, loss_va_ib_contrastive
        

    def training_step(self, batch, batch_idx, dataloader_idx=0, custom_device=None):
        video_frame, video_id, _, ib_video, ib_audio = self.get_input(batch, self.use_cache_video_feat)
        batch_size = video_frame.shape[0]
        device = video_frame.device
        
        # 1. Encode video frame to CLIP embedding.
        if self.use_cache_video_feat:
            video_feat = video_frame                        # [B, F, C]
        else:
            video_feat = self.encode_image(video_frame)     # [B, F, C]
            video_feat = video_feat.detach()
        
        # 2. Forward VAE.
        video_feat = einops.rearrange(video_feat, "b f c -> b c f")
        video_feat_reconstructed, loss_kl = self.video_vae.forward(video_feat)
        
        video_feat = einops.rearrange(video_feat, "b c f -> b f c")
        video_feat_reconstructed = einops.rearrange(video_feat_reconstructed, "b c f -> b f c")
        
        # 3. Compute loss.
        loss_mse, loss_image_contrastive, loss_video_clip_contrastive, \
            loss_video_contrastive, loss_vv_ib_contrastive, loss_va_ib_contrastive \
                = self.compute_losses(video_feat, video_feat_reconstructed, ib_video, ib_audio)
        
        loss = loss_mse \
            + self.loss_kl_weight * loss_kl \
            + self.loss_image_contrastive_weight * loss_image_contrastive \
            + self.loss_video_clip_contrastive_weight * loss_video_clip_contrastive \
            + self.loss_video_contrastive_weight * loss_video_contrastive \
            + self.loss_vv_ib_contrastive_weight * loss_vv_ib_contrastive \
            + self.loss_va_ib_contrastive_weight * loss_va_ib_contrastive
            
        # Loss nan check
        has_nan_loss = torch.isnan(loss).any().item()
        assert not has_nan_loss, "Loss has nan value!"
        
        ''' 7. Log '''
        self.log("train/loss", loss, batch_size=batch_size, sync_dist=True, 
                 on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        self.log("train/loss_mse", loss_mse, batch_size=batch_size, sync_dist=True, 
                 on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        self.log("train/loss_kl", loss_kl, batch_size=batch_size, sync_dist=True,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        self.log("train/loss_image_contrastive", loss_image_contrastive, batch_size=batch_size, sync_dist=True,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        self.log("train/loss_video_clip_contrastive", loss_video_clip_contrastive, batch_size=batch_size, sync_dist=True,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        self.log("train/loss_video_contrastive", loss_video_contrastive, batch_size=batch_size, sync_dist=True,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        self.log("train/loss_vv_ib_contrastive", loss_vv_ib_contrastive, batch_size=batch_size, sync_dist=True,
                 on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        self.log("train/loss_va_ib_contrastive", loss_va_ib_contrastive, batch_size=batch_size, sync_dist=True,
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
        video_frame, video_id, _, ib_video, ib_audio = self.get_input(batch, self.use_cache_video_feat)
        batch_size = video_frame.shape[0]
        
        # 1. Encode video frame to CLIP embedding.
        if self.use_cache_video_feat:
            video_feat = video_frame                        # [B, F, C]
        else:
            video_feat = self.encode_image(video_frame)     # [B, F, C]
            video_feat = video_feat.detach()

        # NOTE Temporally code. For saving video feature.
        video_feat_np = video_feat.cpu().numpy()
        for i, vid in enumerate(video_id):
            np.save(os.path.join(self.val_log_dir_for_video_per_epoch, f"{vid}.npy"), video_feat_np[i])
        return [0, 0]
        
        # 2. Forward VAE.
        video_feat = einops.rearrange(video_feat, "b f c -> b c f")
        video_latent = self.video_vae.encode(video_feat)
        video_latent_sampled = video_latent.sample()
        video_feat_reconstructed = self.video_vae.decode(video_latent_sampled)
        
        video_feat = einops.rearrange(video_feat, "b c f -> b f c")
        video_feat_reconstructed = einops.rearrange(video_feat_reconstructed, "b c f -> b f c")
        
        loss_kl = video_latent.kl().mean()
        
        # 3. Compute loss.
        loss_mse, loss_image_contrastive, loss_video_clip_contrastive, \
            loss_video_contrastive, loss_vv_ib_contrastive, loss_va_ib_contrastive \
                = self.compute_losses(video_feat, video_feat_reconstructed, ib_video, ib_audio)
        
        return [loss_mse, loss_kl, loss_image_contrastive, loss_video_clip_contrastive, loss_video_contrastive, loss_vv_ib_contrastive, loss_va_ib_contrastive]
    
    def validation_epoch_end(self, val_step_outputs):
        # Compute average loss.
        val_loss_mse = torch.stack([x[0] for x in val_step_outputs]).mean()
        val_loss_kl = torch.stack([x[1] for x in val_step_outputs]).mean()
        val_loss_image_contrastive = torch.stack([x[2] for x in val_step_outputs]).mean()
        val_loss_video_clip_contrastive = torch.stack([x[3] for x in val_step_outputs]).mean()
        val_loss_video_contrastive = torch.stack([x[4] for x in val_step_outputs]).mean()
        val_loss_vv_ib_contrastive = torch.stack([x[5] for x in val_step_outputs]).mean()
        val_loss_va_ib_contrastive = torch.stack([x[6] for x in val_step_outputs]).mean()
        
        self.log("val/loss_mse", val_loss_mse, sync_dist=True, prog_bar=True, logger=True,)
        self.log("val/loss_kl", val_loss_kl, sync_dist=True, prog_bar=True, logger=True,)
        self.log("val/loss_image_contrastive", val_loss_image_contrastive, sync_dist=True, prog_bar=True, logger=True,)
        self.log("val/loss_video_clip_contrastive", val_loss_video_clip_contrastive, sync_dist=True, prog_bar=True, logger=True,)
        self.log("val/loss_video_contrastive", val_loss_video_contrastive, sync_dist=True, prog_bar=True, logger=True,)
        self.log("val/loss_vv_ib_contrastive", val_loss_vv_ib_contrastive, sync_dist=True, prog_bar=True, logger=True,)
        self.log("val/loss_va_ib_contrastive", val_loss_va_ib_contrastive, sync_dist=True, prog_bar=True, logger=True,)
        
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

        # NOTE Temporally code. For extracting feature.
        x = x[:, 0, :]
        return einops.rearrange(x, "(b f) c -> b f c", b=_b)

        x = self.image_encoder.visual.ln_post(x[:, 0, :])   # Get CLS token only
        if use_projection:
            if self.image_encoder.visual.proj is not None:
                x = x @ self.image_encoder.visual.proj
        x = einops.rearrange(x, "(b f) c -> b f c", b=_b)
        
        return x
    
    def select_random_segments_with_gradients(self, data, segment_length=10, num_segments=3):
        """
        Args:
            data            : [batch_size, frame_length, dim]
            segment_length  : int
            num_segments    : int
        
        Returns:
            segments        : [batch_size, num_segments, segment_length, dim]
        """
        batch_size, frame_length, dim = data.shape

        if frame_length < segment_length:
            raise ValueError("frame_length should be larger than segment_length")

        start_indices = torch.randint(
            0, frame_length - segment_length + 1, size=(batch_size, num_segments), device=data.device
        )
        segment_indices = start_indices.unsqueeze(-1) + torch.arange(segment_length, device=data.device).unsqueeze(0)
        segment_indices = segment_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
        segments = data.unsqueeze(1).gather(dim=2, index=segment_indices)

        return segments
    
    def select_random_segments_with_gradients(self, data, segment_length=10, num_segments=3):
        """
        Args:
            data            : [batch_size, frame_length, dim]
            segment_length  : int
            num_segments    : int
        
        Returns:
            segments        : [batch_size, num_segments, segment_length, dim]
        """
        batch_size, frame_length, dim = data.shape

        if frame_length < segment_length:
            raise ValueError("frame_length should be larger than segment_length")

        start_indices = torch.randint(
            0, frame_length - segment_length + 1, size=(batch_size, num_segments), device=data.device
        )
        segment_indices = start_indices.unsqueeze(-1) + torch.arange(segment_length, device=data.device).unsqueeze(0)
        # segment_indices: [batch_size, num_segments, segment_length]
        segment_indices = segment_indices.unsqueeze(-1).expand(-1, -1, -1, dim)
        # segment_indices: [batch_size, num_segments, segment_length, dim]
        data = data.unsqueeze(1).expand(-1, num_segments, -1, -1)
        # data: [batch_size, num_segments, frame_length, dim]
        segments = data.gather(dim=2, index=segment_indices)
        # segments: [batch_size, num_segments, segment_length, dim]

        return segments
    
    def info_nce_loss(self, f1, f2, temperature=0.7):
        assert f1.shape == f2.shape
        assert f1.dim() == 2 and f2.dim() == 2
        
        f1 = nn_func.normalize(f1, dim=1)
        f2 = nn_func.normalize(f2, dim=1)
        
        similarity_matrix_12 = torch.matmul(f1, f2.T)  # [batch_size, batch_size]
        similarity_matrix_21 = torch.matmul(f2, f1.T)  # [batch_size, batch_size]
        
        batch_size = f1.shape[0]
        labels = torch.arange(batch_size).to(f1.device)
        
        similarity_matrix_12 = similarity_matrix_12 / temperature
        similarity_matrix_21 = similarity_matrix_21 / temperature
        
        loss_12 = nn_func.cross_entropy(similarity_matrix_12, labels)
        loss_21 = nn_func.cross_entropy(similarity_matrix_21, labels)
        
        loss = 0.5 * (loss_12 + loss_21)
        
        return loss
    
