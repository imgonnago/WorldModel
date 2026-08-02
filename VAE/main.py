import torch
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.VAE import VAE
from Data.Dataset import ImageDataset

print('Dataloading...')
print('\n')
#데이터 로딩
images = np.load('./data/obs.npy')  # (N, 96, 96, 3)
dataset = ImageDataset(images)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
print('Dataloading complete.')
print('\n')

print('VAE Model loading...')
print('\n')
#모델 로딩
vae = VAE(latent_dim = config.LATENT_DIM, flatten_dim = config.FLATTEN_DIM)
print('Model loading complete.')
print('\n')


