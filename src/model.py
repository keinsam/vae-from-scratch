import torch
from torch import nn

class VAE(nn.Module):

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_logvar = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, input_dim)

    def encoder(self, x):
        pass

    def decoder(self, z):
        pass

    def forward(self, x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        x_recon = self.decoder(z)
        return x_recon, mu, sigma

    def reparameterize(self, mu, sigma):
        #std = torch.exp(0.5 * log_sigma)
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon * sigma
        return z
    
    def elbo_loss(self, recon_x, x, mu, sigma):
        recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return recon_loss + kl_divergence