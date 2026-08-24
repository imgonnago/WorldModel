import Vision.v_config as v_config
import torch.nn as nn
import torch
# 3*96*96 -> 16*48*48 -> 32*24*24 -> 64*12*12 -> 128*6*6 -> 4608(flatten_dim)
class Encoder(nn.Module):
  def __init__(self, flatten_dim : int = v_config.FLATTEN_DIM, output_dim: int = v_config.LATENT_DIM, ) -> None:
    super().__init__()

    self.block1 = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
    )

    self.block2 = nn.Sequential(
        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        nn.Dropout(0.2),
    )

    self.block3 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
    )

    self.block4 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.GELU(),
        nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
    )

    self.flatten = nn.Flatten()
    #z값을 분포로 압축하기 위해 선형 레이어를 통해 평균과 분산을 계산
    #평균
    self.fc_mu = nn.Linear(flatten_dim, output_dim)
    #분산
    self.fc_logvar = nn.Linear(flatten_dim, output_dim)

  def forward(self, x):
    #U-net 구조를 위한 forward 구조 
    conv1 = self.block1(x)
    conv2 = self.block2(conv1)
    conv3 = self.block3(conv2)
    conv4 = self.block4(conv3)

    flatten = self.flatten(conv4)
    mu = self.fc_mu(flatten)
    logvar = self.fc_logvar(flatten)

    return mu, logvar, conv1, conv2, conv3