# --------------------------------------------------------
# References:
# GLIDE:    https://github.com/openai/glide-text2im
# MAE:      https://github.com/facebookresearch/mae/blob/main/models_mae.py
# DiT:      https://github.com/facebookresearch/DiT
# --------------------------------------------------------
import math
import numpy as np
import collections.abc
from itertools import repeat
from functools import partial
from typing import Tuple, Union

import torch
import torch.nn as nn

try:
    from torch import _assert
except ImportError:
    def _assert(condition: bool, message: str):
        assert condition, message

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# --------------------------------------------------------------------------
# Sine/Cosine Positional Embedding Functions              
# --------------------------------------------------------------------------
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

# Copied from Meta DiT
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

# Copied from Meta JEPA
def get_1d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid length
    returns:
        pos_embed: [grid_size, embed_dim]   (w/o cls_token)
               or  [1+grid_size, embed_dim] (w/ cls_token)
    """
    grid = np.arange(grid_size, dtype=np.float32)
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# --------------------------------------------------------------------------
# Timesteps Embedding Layers, Patchify, and MLP
# --------------------------------------------------------------------------

# Copied From Meta DiT
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# Copied timm.models.vision_transformer
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


# Copied from timm.models.vision_transformer
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# --------------------------------------------------------------------------
# Core DiT Model
# --------------------------------------------------------------------------

# Modified from Meta DiT
class DiTBlock(nn.Module):
    """
    A DiT block with cross-attention conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1      = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_self  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2      = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_cross = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm3      = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim  = int(hidden_size * mlp_ratio)
        approx_gelu     = lambda: nn.GELU(approximate="tanh")
        self.mlp        = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.)
        
    def forward(self, x: torch.Tensor, c: torch.Tensor):
        x_norm  = self.norm1(x)
        x       = x + self.attn_self(x_norm, x_norm, x_norm, need_weights=False, attn_mask=None)[0]
        x_norm  = self.norm2(x)
        x       = x + self.attn_cross(x_norm, c, c, need_weights=False, attn_mask=None)[0]
        x       = x + self.mlp(self.norm3(x))
        
        return x


# Modified from Meta DiT
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size: int, patch_size: Union[Tuple[int, int], int], out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        linear_out_dim = patch_size[0] * patch_size[1] * out_channels if isinstance(patch_size, tuple) else patch_size ** 2 * out_channels
        self.linear = nn.Linear(hidden_size, linear_out_dim, bias=True)
        
    def forward(self, x: torch.Tensor):
        x = self.linear(self.norm_final(x))

        return x


# Modified from Meta DiT
class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size  : Union[Tuple[int, int], int] = [256, 16],
        patch_size  : Union[Tuple[int, int], int] = 4,
        in_channels : int = 8,
        hidden_size : int = 1152,
        depth       : int = 28,
        num_heads   : int = 16,
        mlp_ratio   : float = 4.0,
        learn_sigma : bool = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # NOTE:
        # No additional linear layer before cross atten here. 
        # Pos Embed will use fixed sin-cos embedding in init.
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.MultiheadAttention):
                torch.nn.init.xavier_uniform_(module.in_proj_weight)
                torch.nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out cross attention layers in DiT blocks: TODO
        for block in self.blocks:
            nn.init.constant_(block.attn_cross.in_proj_weight , 0)
            nn.init.constant_(block.attn_cross.out_proj.weight , 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor):
        """
        x   : (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if isinstance(self.input_size, int):
            h = w = int((self.input_size/p) ** 0.5)
        else:
            h, w = int(self.input_size[0]/p) , int(self.input_size[1]/p)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, C') tensor of conditions
        """
        # Check t
        if len(t.shape) == 0:
            t = t[None]
            t = t.expand(x.shape[0])
        # Forward
        x = self.x_embedder(x) + self.pos_embed     # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                      # (N, D)
        x = torch.cat([t.unsqueeze(1), x], dim=1)   # (N, 1+T, D)
        for block in self.blocks:
            x = block(x, y)                         # (N, 1+T, D)
        x = x[:, 1:, ...]                           # (N, T, D)
        x = self.final_layer(x)                     # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                      # (N, out_channels, H, W)
        
        return x
    
    
# --------------------------------------------------------------------------
# Core DiT Model (AdaLN for Timestep Embedding)
# --------------------------------------------------------------------------

# Modified from Meta DiT
class DiTBlockAdaLN(nn.Module):
    """
    A DiT block with cross-attention conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1      = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_self  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2      = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_cross = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm3      = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim  = int(hidden_size * mlp_ratio)
        approx_gelu     = lambda: nn.GELU(approximate="tanh")
        self.mlp        = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_crs, scale_crs, gate_crs, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(t).chunk(9, dim=1)
        x_adaln = modulate(self.norm1(x), shift_msa, scale_msa)
        x       = x + gate_msa.unsqueeze(1) * self.attn_self(x_adaln, x_adaln, x_adaln, need_weights=False, attn_mask=None)[0]
        x_adaln = modulate(self.norm2(x), shift_crs, scale_crs)
        x       = x + gate_crs.unsqueeze(1) * self.attn_cross(x_adaln, c, c, need_weights=False, attn_mask=None)[0]
        x_adaln = modulate(self.norm3(x), shift_mlp, scale_mlp)
        x       = x + gate_mlp.unsqueeze(1) * self.mlp(x_adaln)
        
        return x


# Modified from Meta DiT
class FinalLayerAdaLN(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size: int, patch_size: Union[Tuple[int, int], int], out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        linear_out_dim = patch_size[0] * patch_size[1] * out_channels if isinstance(patch_size, tuple) else patch_size ** 2 * out_channels
        self.linear = nn.Linear(hidden_size, linear_out_dim, bias=True)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift, scale = self.adaln_modulation(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)

        return x


# Modified from Meta DiT
class DiTAdaLN(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size  : Union[Tuple[int, int], int] = [256, 16],
        patch_size  : Union[Tuple[int, int], int] = 4,
        in_channels : int = 8,
        hidden_size : int = 1152,
        depth       : int = 28,
        num_heads   : int = 16,
        mlp_ratio   : float = 4.0,
        learn_sigma : bool = False,
        do_patchify : bool = True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.do_patchify = do_patchify
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        if self.do_patchify:
            self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
            # Pos Embed will use fixed sin-cos embedding in init.
            num_patches = self.x_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        else:
            assert isinstance(self.input_size, int)
            self.x_embedder = nn.Identity()
            self.pos_embed = nn.Parameter(torch.zeros(1, input_size, hidden_size), requires_grad=False)
            
        # NOTE:
        # No additional linear layer before cross atten here. 
        
        self.blocks = nn.ModuleList([
            DiTBlockAdaLN(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayerAdaLN(hidden_size, patch_size, self.out_channels)
        if not self.do_patchify:
            self.final_layer.linear = nn.Identity()
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.MultiheadAttention):
                torch.nn.init.xavier_uniform_(module.in_proj_weight)
                torch.nn.init.xavier_uniform_(module.out_proj.weight)
                if module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0)
                if module.out_proj.bias is not None:
                    nn.init.constant_(module.out_proj.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.do_patchify:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        else:
            pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.input_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        if self.do_patchify:
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out cross attention layers in DiT blocks: TODO
        for block in self.blocks:
            nn.init.constant_(block.attn_cross.in_proj_weight , 0)
            nn.init.constant_(block.attn_cross.out_proj.weight , 0)

        # Zero-out output layers:
        if self.do_patchify:
            nn.init.constant_(self.final_layer.linear.weight, 0)
            nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor):
        """
        x   : (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        if isinstance(self.input_size, int):
            h = w = int((self.input_size/p) ** 0.5)
        else:
            h, w = int(self.input_size[0]/p) , int(self.input_size[1]/p)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        
        return imgs

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, C') tensor of conditions
        """
        if len(t.shape) == 0:
            t = t[None]
            t = t.expand(x.shape[0])
            
        x = self.x_embedder(x) + self.pos_embed     # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                      # (N, D)
        for block in self.blocks:
            x = block(x, t, y)                      # (N, T, D)
        x = self.final_layer(x, t)                  # (N, T, patch_size ** 2 * out_channels)
        if self.do_patchify:
            x = self.unpatchify(x)                  # (N, out_channels, H, W)
        
        return x


# --------------------------------------------------------------------------
# DiT Configs  
# --------------------------------------------------------------------------

# Base, cross attention to condition, concat timestep
def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

# Adaln, cross attention to condition, adaln for timestep
def DiTAdaLN_XL_2(**kwargs):
    return DiTAdaLN(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiTAdaLN_XL_4(**kwargs):
    return DiTAdaLN(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiTAdaLN_XL_8(**kwargs):
    return DiTAdaLN(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiTAdaLN_L_2(**kwargs):
    return DiTAdaLN(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiTAdaLN_L_4(**kwargs):
    return DiTAdaLN(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiTAdaLN_L_8(**kwargs):
    return DiTAdaLN(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiTAdaLN_B_2(**kwargs):
    return DiTAdaLN(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiTAdaLN_B_4(**kwargs):
    return DiTAdaLN(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiTAdaLN_B_8(**kwargs):
    return DiTAdaLN(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiTAdaLN_S_2(**kwargs):
    return DiTAdaLN(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiTAdaLN_S_4(**kwargs):
    return DiTAdaLN(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiTAdaLN_S_8(**kwargs):
    return DiTAdaLN(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}


if __name__ == '__main__':
    model = DiT_XL_2(input_size=[128,8], in_channels=4, )
    x = torch.randn(2, 4, 128, 8)
    t = torch.randint(0, 100, (2,))
    c = torch.randn(2, 16, 1152)
    print(x.shape, t.shape, c.shape)
    y = model(x, t, c)
    print(y.shape)
    