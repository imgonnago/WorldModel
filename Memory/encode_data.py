import numpy as np
import torch 
import sys
import os
from Vision.v_model.VAE import VAE
import Memory.m_config as m_config

def encode_data(obs_path=m_config.OBS_TRAIN_DIR, 
                next_obs_path=m_config.NEXT_OBS_TRAIN_DIR, 
                vae_ckpt=m_config.CKPT_DIR, 
                save_dir=m_config.SAVE_DIR, 
                device=m_config.DEVICE, 
                batch_size = m_config.BATCH_SIZE):

    os.makedirs(save_dir, exist_ok=True)
    obs = np.load(obs_path)
    next_obs = np.load(next_obs_path)

    vae = VAE(m_config.LATENT_DIM, m_config.FLATTEN_DIM)
    ckpt = torch.load(vae_ckpt, map_location=device)
    vae.load_state_dict(ckpt["model_state_dict"])
    vae.to(device)
    vae.eval()

    def to_z(images):
        z_list=[]
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                batch = torch.tensor(batch, dtype=torch.float32) / 255.0
                batch = batch.permute(0,3,1,2).to(device)
                mu, logvar = vae.encoder(batch)
                z_list.append(mu.cpu().numpy())
        return np.concatenate(z_list, axis=0)

    z = to_z(obs)
    next_z = to_z(next_obs)

    np.save(f"{save_dir}/z.npy",z)
    np.save(f"{save_dir}/next_z.npy",next_z)

    if save_dir == "/z_data/train":
        print(f"train 인코딩 완료: z shape {z.shape}, next_z shape {next_z.shape}")
    if save_dir == "./z_data/val":
        print(f"val 인코딩 완료: z shape {z.shape}, next_z shape {next_z.shape}")

if __name__ == "__main__":

    encode_data(
        obs_path=m_config.OBS_TRAIN_DIR,
        next_obs_path=m_config.NEXT_OBS_TRAIN_DIR,
        vae_ckpt=m_config.CKPT_DIR,
        save_dir=m_config.Z_TRAIN_DIR,
        device=m_config.DEVICE,
    )

    encode_data(
        obs_path=m_config.OBS_VAL_DIR,
        next_obs_path=m_config.NEXT_OBS_VAL_DIR,
        vae_ckpt=m_config.CKPT_DIR,
        save_dir=m_config.Z_VAL_DIR,
        device=m_config.DEVICE,
    )