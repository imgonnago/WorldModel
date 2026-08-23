import torch
import torch.nn as nn
import Vision.v_config as v_config
# 4608(flatten_dim) -> 128*6*6 -> 64*12*12 -> 32*24*24 -> 16*48*48 -> 3*96*96
# Conv2d로 줄였던 채널을 다시 ConvTranspose2d로 늘려서 원래 이미지 크기로 복원
class Decoder(nn.Module):
  def __init__(self, latent_dim: int = v_config.LATENT_DIM, flatten_dim: int = v_config.FLATTEN_DIM) -> None:
    super().__init__()

    self.de_block1 = nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.GELU(),
    )

    self.de_block2 = nn.Sequential(
        nn.ConvTranspose2d(128, 32, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.GELU(),
    )

    self.de_block3 = nn.Sequential(
        nn.ConvTranspose2d(64, 16, kernel_size=4, stride=2, padding=1),
        nn.GELU(),
    )

    self.de_block4 = nn.Sequential(
        nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
        nn.Sigmoid()
    )

    # latent_dim인 z값을 flatten_dim으로 변환하기 위해 선형 레이어를 추가
    self.fc_linear = nn.Linear(latent_dim, flatten_dim)
  
  def forward(self, z, conv1, conv2, conv3):
    x = self.fc_linear(z)
    # flatten_dim인 4608을 128*6*6으로 변환
    x = x.view(-1, 128, 6, 6)
    #인코더에서 출력 차원이 같은 블록끼리 torch.cat
    x = self.de_block1(x)
    x = torch.cat([x, conv3], dim=1)

    x = self.de_block2(x)
    x = torch.cat([x, conv2], dim=1)

    x = self.de_block3(x)
    x = torch.cat([x, conv1], dim=1)

    x = self.de_block4(x)
    # decoder의 출력은 3*96*96이므로, 최종적으로 원래 이미지 크기로 복원됨

    return x