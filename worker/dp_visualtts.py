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

from model.dp.attn_dp import AttnDP_visualtts
from .base import instantiate_from_config, get_class_from_config






class DurationPredictor(pl.LightningModule):
    def __init__(self, 
                 dp_config       : Dict = None,
                 dp_ckpt_path    : str = None,
                 resume_training : bool = False,
                 lr_warmup_steps : int = 2000,
                 ignore_keys     : list = list(),
                 log_data_time   : bool = True):
        super(DurationPredictor, self).__init__()

        self.dp = AttnDP_visualtts(**dp_config)
        self.resume_training = resume_training
        self.dp_ckpt_path = dp_ckpt_path
        if dp_ckpt_path is not None:
            self.load_dp_ckpt(dp_ckpt_path, ignore_keys=ignore_keys)
        self.lr_warmup_steps = lr_warmup_steps
        self.log_data_time = log_data_time
        if self.log_data_time:
            self.last_log_data_time = time.time()


    def load_dp_ckpt(self, dp_ckpt_path: str, ignore_keys=list()):

        ckpt = torch.load(dp_ckpt_path, map_location="cpu")
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
        print(f"=> Restored vaflow ckpt from {dp_ckpt_path}")




    def configure_optimizers(self):
        # LR.
        lr = self.learning_rate

        # Optimizer.
        optimizer = torch.optim.AdamW( 
            filter(lambda p: p.requires_grad, self.parameters()), 
            lr=lr, 
        )
        if self.dp_ckpt_path is not None and self.resume_training:
            ckpt = torch.load(self.dp_ckpt_path, map_location="cpu", weights_only=True)
            optimizer.load_state_dict(ckpt['optimizer_states'][0])
            print(f"=> Restored optimizer from {self.vaflow_ckpt_path}")

            
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
        if self.dp_ckpt_path is not None and self.resume_training:
            ckpt = torch.load(self.dp_ckpt_path, map_location="cpu", weights_only=True)
            scheduler["scheduler"].load_state_dict(ckpt['lr_schedulers'][0])
            print(f"=> Restored scheduler from {self.vaflow_ckpt_path}")


        return [optimizer], [scheduler]
    

    def loss_function_attn(self, pred, target, avhubert_length, phone_length):
        target_resized = copy.deepcopy(target)
        for i in range(len(pred)):
            if avhubert_length is not None:
                pred[i, avhubert_length[i]:] = 0.0   # [250, max_pho_len]
                target_resized[i, :avhubert_length[i]] = target_resized[i, :avhubert_length[i]] / target_resized[i, :avhubert_length[i]].sum(-1,keepdim=True)  # [250, max_pho_len]

        # # l1_loss    smooth_l1_loss    mse_loss
        loss = nn_func.smooth_l1_loss(pred, target_resized, beta = 0.0005)  

        # true_prob = (pred * target).sum(-1)  # [bs, 250]
        # for i in range(len(true_prob)):
        #     true_prob[i, avhubert_length[i]:] = 1
        # true_prob = true_prob.reshape(-1,1)  # [bs * 250, 1]
        # input = torch.concat([true_prob, 1-true_prob], dim = -1)
        # target = torch.zeros_like(true_prob.squeeze(-1), dtype=torch.int64)
        # loss = nn_func.cross_entropy(input, target)
        return loss
    

    def loss_function_span(self, pred, target, avhubert_length, phone_length):
        for i in range(len(pred)):
            if avhubert_length is not None:
                pred[i, avhubert_length[i]:] = 0.0

        loss = nn_func.mse_loss(pred, target)
        return loss
    

    def get_input(self, batch, use_cache_video_feat=False):

        # Base input
        video_id, duration_matrix, duration_span, avhubert, avhubert_length, phone_id, phone_length = batch['video_id'], batch['duration_matrix'], batch['duration_span'], batch['avhubert'], batch['avhubert_length'], batch['phone_id'], batch['phone_length']
        # phone_id = phone_id.unsqueeze(-1)  # [bs, phone_seq_len, 1]
        # Type convert
        duration_span = duration_span.float()

        return video_id, duration_matrix, duration_span, avhubert, avhubert_length, phone_id, phone_length
    


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
        #          [bs, 250, max_pho_len]  [bs, max_pho_len, 2]    [bs, 250, 1024]    10*[audio_len]    [bs, max_pho_len]  10*[pho_len]
        video_id,  duration_matrix,         duration_span,          avhubert,         avhubert_length,   phone_id,          phone_length = self.get_input(batch)
        batch_size = duration_matrix.shape[0]

        # Forward
        # [bs, max_pho_len, 2]   list()
        duration_pred,           attn_weight_list = self.dp(phone_id, avhubert)
        # [bs, 250, max_pho_len]
        final_attn_weight = attn_weight_list[-1]

        # GT
        start_offset = copy.deepcopy(duration_span[...,0]) # [bs, 250]
        start_offset[:,1:] = duration_span[:,1:,0] - duration_span[:,:-1,1] + 1  # [bs, 250]
        length = duration_span[...,1] - duration_span[...,0]  # [bs, 250]
        duration_gt = torch.stack([start_offset, length], dim=-1)  # [bs, 250, 2]

        # Loss
        loss1 = self.loss_function_attn(final_attn_weight, duration_matrix, avhubert_length, phone_length)
        loss2 = self.loss_function_span(duration_pred, duration_gt, avhubert_length, phone_length)
        loss = loss1
        # loss = loss1 * ((loss1.detach() + loss2.detach())/loss1.detach()) + loss2 * ((loss1.detach() + loss2.detach())/loss2.detach())


        # Log
        self.log("train/loss", loss, batch_size=batch_size, sync_dist=True, 
                    on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        return loss

    
    def on_train_batch_end(self, batch, batch_idx, dataloader_idx=0):
        if self.log_data_time:
            self.last_log_data_time = time.time()




    def validation_step_(self, batch, batch_idx, dataloader_idx=0, custom_device=None):
        #          [bs, 250, max_pho_len]  [bs, max_pho_len, 2]    [bs, 250, 1024]    10*[audio_len]    [bs, max_pho_len]  10*[pho_len]
        video_id,  duration_matrix,         duration_span,          avhubert,         avhubert_length,   phone_id,          phone_length = self.get_input(batch)
        batch_size = duration_matrix.shape[0]

        # Forward
        # [bs, max_pho_len, 2]   list()
        duration_pred,           attn_weight_list = self.dp(phone_id, avhubert)
        # [bs, 250, max_pho_len]
        final_attn_weight = attn_weight_list[-1]

        # GT
        start_offset = copy.deepcopy(duration_span[...,0]) # [bs, 250]
        start_offset[:,1:] = duration_span[:,1:,0] - duration_span[:,:-1,1] + 1  # [bs, 250]
        length = duration_span[...,1] - duration_span[...,0]  # [bs, 250]
        duration_gt = torch.stack([start_offset, length], dim=-1)  # [bs, 250, 2]

        # Loss
        loss1 = self.loss_function_attn(final_attn_weight, duration_matrix, avhubert_length, phone_length)
        loss2 = self.loss_function_span(duration_pred, duration_gt, avhubert_length, phone_length)
        loss = loss1
        # loss = loss1 * ((loss1.detach() + loss2.detach())/loss1.detach()) + loss2 * ((loss1.detach() + loss2.detach())/loss2.detach())


        # Log
        self.log("val/loss", loss, batch_size=batch_size, sync_dist=True, 
                    on_step=True, on_epoch=True, prog_bar=True, logger=True,)
        
        for i, audio in enumerate(final_attn_weight):
            aw = final_attn_weight[i].cpu().numpy()  # [250, max_pho_len]
            aw = aw[:,:phone_length[i]]  # [250, pho_len]
            aw[avhubert_length[i]:] = 0  # [250, pho_len]
            audio_path = os.path.join(self.val_log_dir_for_video_per_epoch, "{}_attn".format(video_id[i]))
            np.save(audio_path, aw)
            
            
            # # predicted_length = durat    ion_pred[i, :avhubert_length[i],1] # [pho_len]
            # # predicted_length = torch.where(predicted_length < 1, torch.ones_like(predicted_length), predicted_length)
            # # predicted_offset = duration_pred[i, :avhubert_length[i],0] # [pho_len]
            # # predicted_offset = (predicted_offset >= 0.5).int()          # [pho_len]
            # # ## GT
            # # # predicted_length = length[i,:avhubert_length[i]]
            # # # predicted_offset = start_offset[i,:avhubert_length[i]]
            # # ## Scale
            # # # predicted_total_phone_length = (predicted_length + predicted_offset - 1).sum()
            # # scale = ((phone_length[i] + avhubert_length[i])-predicted_offset.sum()) / predicted_length.sum()
            # # predicted_length = predicted_length * scale

            # # aw = torch.zeros_like(torch.from_numpy(aw))
            # # current_pos = 0
            # # for hubert_id in range(avhubert_length[i]):
            # #     offset = predicted_offset[hubert_id].item()
            # #     # offset = int(round(offset, 0))

            # #     duration = predicted_length[hubert_id].item()
            # #     duration = min(phone_length[i] - current_pos, duration)

            # #     aw[hubert_id, int(current_pos + offset): int(current_pos + offset + duration)] = 1
            # #     current_pos = current_pos + offset + duration - 1
            # # audio_path = os.path.join(self.val_log_dir_for_video_per_epoch, "{}".format(video_id[i]))
            np.save(audio_path, aw)
            


    """ Validation """
    def on_validation_epoch_start(self):
        # Set dirs
        self.val_log_dir = os.path.join(self.trainer.logger.save_dir, "val")
        os.makedirs(self.val_log_dir, exist_ok=True)
        self.val_log_dir_for_video_per_epoch = os.path.join(self.val_log_dir, "video", "epoch_{:04d}_global_step_{:.2e}".format(self.trainer.current_epoch, self.global_step))
        os.makedirs(self.val_log_dir_for_video_per_epoch, exist_ok=True)
    

    def validation_step(self, batch, batch_idx, dataloader_idx=0, custom_device=None):
        with torch.no_grad():
            self.validation_step_(batch=batch, batch_idx=batch_idx)
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
        with torch.no_grad():
            self.validation_step_(batch=batch, batch_idx=batch_idx)
        return 0.0