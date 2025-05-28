import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, channel_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.block1 = self._block(channel_dim, hidden_dim // 4, stride=2)
        self.block2 = self._block(hidden_dim // 4, hidden_dim // 2, stride=2)
        self.block3 = self._block(hidden_dim // 2, hidden_dim, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(4)
        self.flatten = nn.Flatten()
        self.conv_to_mu = nn.Linear(hidden_dim * 4 * 4, latent_dim)
        self.conv_to_sigma = nn.Linear(hidden_dim * 4 * 4, latent_dim)
        
    def _block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        mu = self.conv_to_mu(x)
        sigma = self.conv_to_sigma(x)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, channel_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.latent_to_conv = nn.Linear(latent_dim, hidden_dim * 4 * 4)
        self.block1 = self._block(hidden_dim, hidden_dim // 2, stride=2)
        self.block2 = self._block(hidden_dim // 2, hidden_dim // 4, stride=2)
        self.block3 = self._block(hidden_dim // 4, channel_dim, stride=2)
        self.conv_to_output = nn.Conv2d(channel_dim, channel_dim, 1)
        
    def _block(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, z):
        h = self.latent_to_conv(z)
        h = h.view(-1, self.hidden_dim, 4, 4)
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        x_hat = self.conv_to_output(h)
        return torch.sigmoid(x_hat)


class VAE(nn.Module):
    def __init__(self, channel_dim=3, hidden_dim=64, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(channel_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(channel_dim, hidden_dim, latent_dim)
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_hat = self.decoder(z)
        return x_hat, mu, sigma
    
    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        return mu + epsilon * sigma