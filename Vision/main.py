import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchinfo import summary
import Vision.v_config as v_config
from torch.utils.data import Dataset, DataLoader
from v_model.VAE import VAE
from Data.Dataset import ImageDataset
from train import train_vae
import random

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
summary(vae, input_size=(2,3,96,96))

print('Training VAE...')
print('\n')
trained_vae, trained_loss, val_loss = train_vae(
    vae, train_dataloader, val_dataloader,
    epochs=v_config.EPOCHS, lr=v_config.LR, device=v_config.DEVICE,
    checkpoint_dir=v_config.CHECKPOINT_DIR,
)
print('Training complete!')
print('\n')
print("train loss/val loss graph")

random_num = random.randint(0,100)
plt.plot(trained_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.title("U-net Structure Model Fitting Loss")
plt.xlabel("Epoch")
plt.ylabel("loss") 
plt.legend()
plt.savefig(f"C:/Users/zxfg0/WorldModel/figures/VAEModelLoss{random_num}.png")
plt.show   