from typing import Dict, Optional, Tuple, Union, List

import torch
from torch import nn

from .module import DownUpBlock1D, UNetMidBlock1D
from .util import DiagonalGaussianDistribution, randn_tensor


class VanillaVAE_1D(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, 
                 norm_num_groups: int = 32, decoder_with_mid: bool = True) -> None:
        super(VanillaVAE_1D, self).__init__()
        assert hidden_dims is not None, "Please provide the hidden_dims"
    
        self.latent_dim = latent_dim
        
        # Build encoder.
        modules = []
        
        # In conv.
        modules.append( nn.Conv1d(in_channels, hidden_dims[0], 7, padding=3) )
        
        # Down blocks.
        last_output_channel = hidden_dims[0]
        for i, h_dim in enumerate(hidden_dims):
            modules.append( DownUpBlock1D(out_channels=h_dim, in_channels=last_output_channel) )
            last_output_channel = h_dim
        
        # Mid block and final layer.
        modules.append( UNetMidBlock1D(hidden_dims[-1], hidden_dims[-1], hidden_dims[-1]) )
        modules.append(
            nn.Sequential(
                nn.GroupNorm(num_channels=hidden_dims[-1], num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
                nn.Conv1d(hidden_dims[-1], self.latent_dim * 2, 3, padding=1),
            )
        )
        
        self.encoder = nn.Sequential(*modules)

        # Build decoder.
        modules = []
        
        # In conv.
        modules.append( nn.Conv1d(self.latent_dim, hidden_dims[-1], 3, padding=1) )
        
        # Mid block.
        if decoder_with_mid:
            modules.append( UNetMidBlock1D(hidden_dims[-1], hidden_dims[-1], hidden_dims[-1]) )
        
        # Up blocks.
        reversed_hidden_dims = list(reversed(hidden_dims))
        last_output_channel = reversed_hidden_dims[0]
        for i, h_dim in enumerate(reversed_hidden_dims):
            modules.append( DownUpBlock1D(out_channels=h_dim, in_channels=last_output_channel) )
            last_output_channel = h_dim
            
        # Final layer.
        modules.append(
            nn.Sequential(
                nn.GroupNorm(num_channels=reversed_hidden_dims[-1], num_groups=norm_num_groups, eps=1e-6),
                nn.SiLU(),
                nn.Conv1d(reversed_hidden_dims[-1], in_channels, 5, padding=2),
            )
        )
        
        self.decoder = nn.Sequential(*modules)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        h = self.encoder(x)
        posterior = DiagonalGaussianDistribution(h)

        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        dec = self.decoder(z)
        
        return dec

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        x = sample
        # Encode.
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        var = torch.exp(logvar)
        # KL.
        kl = 0.5 * torch.sum(
            torch.pow(mean, 2) + var - 1.0 - logvar,
            dim=[1, 2,],
        )
        # Sample.
        z_sampled = mean + std * randn_tensor(mean.shape, generator=generator, device=mean.device, dtype=mean.dtype)
        # Decode.
        dec = self.decoder(z_sampled)

        return dec, kl.mean()


if __name__ == "__main__":
    vae = VanillaVAE_1D(in_channels=768, latent_dim=320, hidden_dims=[768, 512, 320])
    x = torch.randn(1, 768, 100)
    z = vae.encode(x)
    z = z.sample()
    x_reconstructed = vae.decode(z)
    print(x.shape)
    print(z.shape)
    print(x_reconstructed.shape)