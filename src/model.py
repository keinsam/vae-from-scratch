import torch
import torch.nn as nn

class VAE(nn.Module):
   def __init__(self, input_dim, hidden_dim, latent_dim):
      
      super(VAE, self).__init__()

      self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
      self.decoder = Decoder(input_dim, hidden_dim, latent_dim)

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decoder(z)
        return x_recon, mu, log_var

    def reparameterize(self, mu, log_var):
       pass

class Encoder(nn.Module):
    pass

class Decoder(nn.Module):
    pass