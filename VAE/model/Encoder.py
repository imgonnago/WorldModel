import config
import torch.nn as nn
import torch

class Encoder(nn.Module):
  def __init__(self, flatten_dim : int = config.FLATTEN_DIM, output_dim: int = config.LATENT_DIM, ) -> None:
    super().__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),

        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Dropout(0.2),

        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Dropout(0.2),

        nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),

        nn.Flatten()
    )

    self.fc_mu = nn.Linear(flatten_dim, output_dim)

    self.fc_logvar = nn.Linear(flatten_dim, output_dim)

  def forward(self, x):
    x = self.encoder(x)
    mu = self.fc_mu(x)
    logvar = self.fc_logvar(x)

    return mu, logvar

    
