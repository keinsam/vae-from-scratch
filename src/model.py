import torch
from torch import nn

class VAE(nn.Module) :

    def __init__(self, input_dim, hidden_dim, latent_dim) :
        super(VAE, self).__init__()
        # Encoder
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)
        self.hidden_to_mu = nn.Linear(hidden_dim, latent_dim)
        self.hidden_to_sigma = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        self.hidden_to_output = nn.Linear(hidden_dim, input_dim)
        # Activation function
        self.relu = nn.ReLU()

    def encode(self, x) :
        h = self.relu(self.input_to_hidden(x))
        mu = self.hidden_to_mu(h)
        sigma = self.hidden_to_sigma(h)
        return mu, sigma

    def decode(self, z) :
        h = self.relu(self.latent_to_hidden(z))
        x_recon = self.hidden_to_output(h)
        return torch.sigmoid(x_recon)

    def forward(self, x) :
        mu, sigma = self.encode(x)
        z = self.reparameterize(mu, sigma)
        x_recon = self.decode(z)
        return x_recon, mu, sigma

    def reparameterize(self, mu, sigma) :
        epsilon = torch.randn_like(sigma)
        z = mu + epsilon * sigma
        return z
    
    def elbo_loss(self, recon_x, x, mu, sigma) :
        recon_loss = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return recon_loss + kl_divergence