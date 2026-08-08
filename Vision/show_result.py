import matplotlib.pyplot as plt
import torch
from Data.Dataset import ImageDataset
from v_model.VAE import VAE
import v_config
import numpy as np

ckpt = torch.load("C:/Users/zxfg0/WorldModel/checkpoints/best_checkpoint.pth", map_location=torch.device("cuda"))
vae = VAE(flatten_dim=v_config.FLATTEN_DIM, latent_dim=v_config.LATENT_DIM)
vae.to(v_config.DEVICE)
val_data = np.load("C:/Users/zxfg0/WorldModel/save_data/val/obs.npy")
val_dataset = ImageDataset(val_data)

vae.load_state_dict(ckpt['model_state_dict'])
vae.eval()

with torch.no_grad():
    sample = val_dataset[0].unsqueeze(0).to(v_config.DEVICE)  # Add batch dimension
    recon, mu, logvar = vae(sample)

fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(sample[0].cpu().permute(1, 2, 0))
axes[0].set_title("Original Image")
axes[1].imshow(recon[0].cpu().permute(1, 2, 0))
axes[1].set_title("Reconstructed Image")
plt.savefig("C:/Users/zxfg0/WorldModel/reconstruction.png")
plt.show()