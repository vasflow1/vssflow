from typing import Tuple, Union

import torch
import torch.nn as nn

from .dit_torch import modulate, Mlp, FinalLayerAdaLN, TimestepEmbedder, get_1d_sincos_pos_embed


# Modified from Meta DiT, remove cross-attention
class DiTBlockAdaLN(nn.Module):
    """
    A DiT block with cross-attention conditioning.
    """
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, **block_kwargs):
        super().__init__()
        self.norm1      = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn_self  = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2      = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim  = int(hidden_size * mlp_ratio)
        approx_gelu     = lambda: nn.GELU(approximate="tanh")
        self.mlp        = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.)
        self.adaln_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor,):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation(t).chunk(6, dim=1)
        x_adaln = modulate(self.norm1(x), shift_msa, scale_msa)
        x       = x + gate_msa.unsqueeze(1) * self.attn_self(x_adaln, x_adaln, x_adaln, need_weights=False, attn_mask=None)[0]
        x_adaln = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x       = x + gate_mlp.unsqueeze(1) * self.mlp(x_adaln)
        
        return x


# Modified from Meta DiT
class DiTAdaLN1D(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels : int = 8,
        in_seq_len  : int = 300,
        hidden_size : int = 256,
        depth       : int = 28,
        num_heads   : int = 16,
        mlp_ratio   : float = 4.0,
        learn_sigma : bool = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.in_seq_len = in_seq_len
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, in_seq_len, hidden_size), requires_grad=False)
            
        self.blocks = nn.ModuleList([
            DiTBlockAdaLN(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayerAdaLN(hidden_size, 1, self.out_channels)
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
        pos_embed = get_1d_sincos_pos_embed(self.hidden_size, self.in_seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor,):
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
            x = block(x, t)                         # (N, T, D)
        x = self.final_layer(x, t)                  # (N, T, D)
        
        return x


# Modified from Meta DiT
class DiTBlockAdaLNCross(nn.Module):
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
class DiTAdaLN1DCross(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        in_channels : int = 8,
        in_seq_len  : int = 300,
        cross_seq_len : int = 300,
        hidden_size : int = 256,
        depth       : int = 28,
        num_heads   : int = 16,
        mlp_ratio   : float = 4.0,
        learn_sigma : bool = False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.in_seq_len = in_seq_len
        self.cross_seq_len = cross_seq_len
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, in_seq_len, hidden_size), requires_grad=False)
        
        self.cross_pos_embed = nn.Parameter(torch.zeros(1, cross_seq_len, hidden_size), requires_grad=False)
            
        self.blocks = nn.ModuleList([
            DiTBlockAdaLNCross(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayerAdaLN(hidden_size, 1, self.out_channels)
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
        pos_embed = get_1d_sincos_pos_embed(self.hidden_size, self.in_seq_len)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        cross_pos_embed = get_1d_sincos_pos_embed(self.hidden_size, self.cross_seq_len)
        self.cross_pos_embed.data.copy_(torch.from_numpy(cross_pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

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

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor):
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
            x = block(x, t, c)                      # (N, T, D)
        x = self.final_layer(x, t)                  # (N, T, patch_size ** 2 * out_channels)
        
        return x