import torch.nn as nn
import torch
import Vision.v_config as v_config
from .Decoder import Decoder
from .Encoder import Encoder

class VAE(nn.Module):
  def __init__(self, latent_dim: int = v_config.LATENT_DIM, flatten_dim: int = v_config.FLATTEN_DIM) -> None:
    super().__init__()
    
    self.encoder = Encoder(flatten_dim, latent_dim)
    self.decoder = Decoder(latent_dim, flatten_dim)
  # reparameterization trick을 통해 z값을 샘플링
  # 평균과 분산을 사용하여 z값을 샘플링하는 과정에서, 평균과 분산을 통해 샘플링된 z값이 정규분포를 따르도록 함
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


    
  