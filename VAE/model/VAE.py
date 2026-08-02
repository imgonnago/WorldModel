import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import os
import config
from model.Decoder import Decoder
from model.Encoder import Encoder

class VAE(nn.Module):
  def __init__(self, latent_dim: int = config.LATENT_DIM, flatten_dim: int = config.FLATTEN_DIM) -> None:
    super().__init__()
    
    self.encoder = Encoder(flatten_dim, latent_dim)
    self.decoder = Decoder(latent_dim, flatten_dim)

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


    
  