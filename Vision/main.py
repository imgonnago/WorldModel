import torch
import numpy as np
import Vision.v_config as v_config
from torch.utils.data import Dataset, DataLoader
from v_model.VAE import VAE
from Data.Dataset import ImageDataset
from train import train_vae

print('Dataloading...')
print('\n')
#데이터 로딩
train_images = np.load('./save_data/train/obs.npy')  # (N, 96, 96, 3)
val_images = np.load('./save_data/val/obs.npy')  # (N, 96, 96, 3)

train_dataset = ImageDataset(train_images)
val_dataset = ImageDataset(val_images)

train_dataloader = DataLoader(train_dataset, batch_size=v_config.BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=v_config.BATCH_SIZE, shuffle=False)

print('Dataloading complete!')
print('\n')

print('VAE Model loading...')
print('\n')
#모델 로딩
vae = VAE(latent_dim = v_config.LATENT_DIM, flatten_dim = v_config.FLATTEN_DIM)
print('Model loading complete.')
print('\n')

print('Training VAE...')
print('\n')
trained_vae = train_vae(
    vae, train_dataloader, val_dataloader,
    epochs=v_config.EPOCHS, lr=v_config.LR, device=v_config.DEVICE,
    checkpoint_dir=v_config.CHECKPOINT_DIR,
)
print('Training complete!')
print('\n')
"""print("show reconstruction...")
print('\n')
show_reconstruction()`
print("show reconstruction complete!")
"""

    