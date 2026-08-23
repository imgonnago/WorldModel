import torch.nn as nn
import torch
# U-Net 적용 이전 구조 (latent_dim=48)
# 3*96*96 -> 16*48*48 -> 32*24*24 -> 64*12*12 -> 128*6*6 -> 4608(flatten_dim)
class old_Encoder(nn.Module):
  def __init__(self, flatten_dim: int = 4608, output_dim: int = 24) -> None:
    super().__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Dropout(0.2),

        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        

        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

        nn.Flatten()
    )

    self.fc_mu = nn.Linear(flatten_dim, output_dim)
    self.fc_logvar = nn.Linear(flatten_dim, output_dim)

  def forward(self, x):
    x = self.encoder(x)
    mu = self.fc_mu(x)
    logvar = self.fc_logvar(x)
    return mu, logvar