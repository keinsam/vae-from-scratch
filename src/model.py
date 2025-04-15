import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_sigma = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h = self.relu(self.input_to_hidden(x))
        mu = self.hidden_to_mu(h)
        sigma = self.hidden_to_sigma(h)
        return mu, sigma


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, z):
        h = self.relu(self.latent_to_hidden(z))
        return torch.sigmoid(self.hidden_to_output(h))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
    
    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, sigma
    
    def reparameterize(self, mu, sigma):
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon * sigma
        return z