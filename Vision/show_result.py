import matplotlib.pyplot as plt
import torch
from Data.Dataset import ImageDataset
from v_model.VAE import VAE
import v_config
import numpy as np
import random

ckpt = torch.load("C:/Users/zxfg0/WorldModel/checkpoints/best_checkpoint.pth", map_location=torch.device("cuda"))
vae = VAE(flatten_dim=v_config.FLATTEN_DIM, latent_dim=v_config.LATENT_DIM)
vae.to(v_config.DEVICE)
val_data = np.load("C:/Users/zxfg0/WorldModel/save_data/val/obs.npy")
val_dataset = ImageDataset(val_data)

vae.load_state_dict(ckpt['model_state_dict'])
vae.eval()

random_idx = random.randint(0, len(val_dataset) - 1)

with torch.no_grad():
    sample = val_dataset[random_idx].unsqueeze(0).to(v_config.DEVICE)  # Add batch dimension
    recon, mu, logvar = vae(sample)

random_num = random.randint(0,500)
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(sample[0].cpu().permute(1, 2, 0))
axes[0].set_title("Original Image")
axes[1].imshow(recon[0].cpu().permute(1, 2, 0))
axes[1].set_title("Reconstructed Image")
plt.savefig(f"C:/Users/zxfg0/WorldModel/figures/reconstruction_u-net_version{random_num}.png")
plt.show()