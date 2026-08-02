from pyexpat import model

import torch
import numpy as np
import config
from torch.utils.data import Dataset, DataLoader
from model.VAE import VAE
from Data.Dataset import ImageDataset
from train import train_vae

print('Dataloading...')
print('\n')
#데이터 로딩
train_images = np.load('./data/train/obs.npy')  # (N, 96, 96, 3)
val_images = np.load('./data/val/obs.npy')  # (N, 96, 96, 3)

train_dataset = ImageDataset(train_images)
val_dataset = ImageDataset(val_images)

train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

print('Dataloading complete!')
print('\n')

print('VAE Model loading...')
print('\n')
#모델 로딩
vae = VAE(latent_dim = config.LATENT_DIM, flatten_dim = config.FLATTEN_DIM)
print('Model loading complete.')
print('\n')

print('Training VAE...')
print('\n')
trained_vae = train_vae(
    vae, train_dataloader, val_dataloader,
    epochs=config.EPOCHS, lr=config.LR, device=config.DEVICE,
    checkpoint_dir=config.CHECKPOINT_DIR,
)
print('Training complete!')
print('\n')


