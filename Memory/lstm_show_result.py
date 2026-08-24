#put obs, next_obs images to encoder and output is obs_z, next_obs_z, conv1,2,3. then obs_z is put into lstm for predicting. 
#lstm's output is predict_z that compared with next_obs. so decoder decode predict_z and next_obs.
#
#the problem was i coudln't use decoder that use U-net structure. because U-net use skip layer so decoder need conv1,2,3 for input. 
#but original code not able to use encoder that's why i used old_decoder not U-net decoder

import numpy as np
import torch
from m_model.LSTM import LSTM
import m_config
from Vision.v_model.Encoder import Encoder
from Vision.v_model.Decoder import Decoder
from Vision import v_config
from ShowDataset import get_show_datasets
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

train_dataset, val_dataset = get_show_datasets()

random_idx = random.randint(0, len(val_dataset) - 1)
obs, action, next_obs = val_dataset[random_idx]

#data
#next_obs is not for encode. only use obs for encode
obs = obs.unsqueeze(0).to("cuda")          
action = action.unsqueeze(0).to("cuda")
next_obs = next_obs.to("cuda")

#transform mu to z 
def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(std)
    z = mu + std * epsilon

    return z

#load Encoder
encoder = Encoder(flatten_dim=v_config.FLATTEN_DIM, output_dim=v_config.LATENT_DIM)
v_checkpoint = torch.load(f'{v_config.CHECKPOINT_DIR}/best_checkpoint.pth', map_location=m_config.DEVICE)
encoder_state_dict = {
    k[len("encoder."):]: v
    for k, v in v_checkpoint["model_state_dict"].items()
    if k.startswith("encoder.")
}
encoder.load_state_dict(encoder_state_dict)
encoder.to("cuda")
encoder.eval()


#load Decoder
decoder = Decoder(latent_dim=v_config.LATENT_DIM, flatten_dim=v_config.FLATTEN_DIM)
v_checkpoint = torch.load(f'{v_config.CHECKPOINT_DIR}/best_checkpoint.pth', map_location=m_config.DEVICE)
decoder_state_dict = {
    k[len("decoder."):]: v
    for k, v in v_checkpoint["model_state_dict"].items()
    if k.startswith("decoder.")
}
decoder.load_state_dict(decoder_state_dict)
decoder.to("cuda")
decoder.eval()


#load LSTM 
lstm = LSTM(
    latent_dim=m_config.LATENT_DIM,
    action_dim=m_config.ACTION_DIM,
    hidden_size=m_config.HIDDEN_SIZE
    )

lstm_checkpoint = torch.load(f"{m_config.CKPT_LSTM_DIR}/best_checkpoint.pth", map_location=m_config.DEVICE)
lstm.load_state_dict(lstm_checkpoint["model_state_dict"])
lstm.to("cuda")
lstm.eval()

#random sequence 
random_seq = random.randint(0, m_config.SEQUENCE_LEN - 1)

#run 
with torch.no_grad():
    obs_flat = obs.squeeze(0)
    mu, logvar, conv1, conv2, conv3 = encoder(obs_flat)
    obs_z = reparameterize(mu, logvar)

    z_seq = mu.unsqueeze(0)                           # (1, 30, latent_dim)
    predicted_next_z = lstm(obs_z, action)            # (1, 30, latent_dim)
    predicted_next_z = predicted_next_z.squeeze(0)    # (30, latent_dim)

    predicted_image = decoder(
        predicted_next_z[random_seq].unsqueeze(0),
        conv1[random_seq].unsqueeze(0),
        conv2[random_seq].unsqueeze(0),
        conv3[random_seq].unsqueeze(0),
    )

    actual_image = next_obs[random_seq].unsqueeze(0)

#visualization 
random_num = random.randint(0,500)
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(actual_image[0].cpu().permute(1, 2, 0))
axes[0].set_title("Actual Next State")
axes[1].imshow(predicted_image[0].cpu().permute(1, 2, 0))
axes[1].set_title("Predicted Next State")
plt.savefig(f"C:/Users/zxfg0/WorldModel/figures/predicted_u-net_version{random_num}.png")
plt.show()