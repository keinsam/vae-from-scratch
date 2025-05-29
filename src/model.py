from typing import Tuple
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self,
                 channel_dim: int,
                 hidden_dim: int,
                 latent_dim: int
                ) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.block1 = self._block(channel_dim, hidden_dim // 4, kernel_size=3, stride=2, padding=1)
        self.block2 = self._block(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1)
        self.block3 = self._block(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)
        self.flatten = nn.Flatten()
        self.conv_to_mu = nn.Linear(hidden_dim * 4 * 4, latent_dim)
        self.conv_to_sigma = nn.Linear(hidden_dim * 4 * 4, latent_dim)
        
    def _block(self,
               in_channels: int,
               out_channels: int,
               kernel_size: int = 3,
               stride: int = 1,
               padding: int = 0
              ) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self,
                x: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        mu = self.conv_to_mu(x)
        sigma = self.conv_to_sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self,
                 channel_dim: int,
                 hidden_dim: int,
                 latent_dim: int
                ) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.latent_to_conv = nn.Linear(latent_dim, hidden_dim * 4 * 4)
        self.block1 = self._block(hidden_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block2 = self._block(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.block3 = self._block(hidden_dim // 4, channel_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_to_output = nn.Sequential(nn.Conv2d(channel_dim, channel_dim, kernel_size=1),
                                            nn.Sigmoid()) # nn.Tanh())
        
    def _block(self,
               in_channels: int,
               out_channels: int,
               kernel_size:int = 3,
               stride: int = 1,
               padding: int = 0,
               output_padding: int = 0
               ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self,
                z: torch.Tensor
                ) -> torch.Tensor:
        h = self.latent_to_conv(z)
        h = h.view(-1, self.hidden_dim, 4, 4)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        x_hat = self.conv_to_output(h)
        return x_hat


class VAE(nn.Module):
    def __init__(self,
                 channel_dim: int = 3,
                 hidden_dim: int = 64,
                 latent_dim: int = 32
                 ) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(channel_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(channel_dim, hidden_dim, latent_dim)
    
    def forward(self,
                x: torch.Tensor
                ) -> torch.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma
    
    def reparameterize(self,
                       mu: torch.Tensor,
                       sigma: torch.Tensor
                       ) -> torch.Tensor:
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma