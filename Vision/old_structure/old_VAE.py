import torch.nn as nn
import torch
from .old_Encoder import old_Encoder
from .old_Decoder import old_Decoder

class old_VAE(nn.Module):
  def __init__(self, latent_dim: int = 24, flatten_dim: int = 4608) -> None:
    super().__init__()
    self.encoder = old_Encoder(flatten_dim, latent_dim)
    self.decoder = old_Decoder(latent_dim, flatten_dim)

  def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    z = mu + std * epsilon
    return z

  def forward(self, x):
    mu, logvar = self.encoder(x)
    z = self.reparameterize(mu, logvar)
    decoded_result = self.decoder(z)
    return decoded_result, mu, logvar