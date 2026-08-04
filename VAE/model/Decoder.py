import torch
import torch.nn as nn
import config

class Decoder(nn.Module):
  def __init__(self, latent_dim: int = config.LATENT_DIM, flatten_dim: int = config.FLATTEN_DIM) -> None:
    super().__init__()

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.GELU(),

        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.GELU(),

        nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
        nn.GELU(),

        nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )

    self.fc_linear = nn.Linear(latent_dim, flatten_dim)
  
  def forward(self, z):
    x = self.fc_linear(z)
    x = x.view(-1, 128, 6, 6)
    x = self.decoder(x)

    return x