import numpy as np
import torch
from m_model.LSTM import LSTM
import m_config
from Vision.old_structure.old_Decoder import old_Decoder
from Vision import v_config
from SequenceDataset import get_datasets
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

train_dataset, val_dataset = get_datasets()

train_loader = DataLoader(train_dataset, batch_size=m_config.TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=m_config.TRAIN_BATCH_SIZE, shuffle=False)

lstm = LSTM(
    latent_dim=m_config.LATENT_DIM,
    action_dim=m_config.ACTION_DIM,
    hidden_size=m_config.HIDDEN_SIZE
    )

lstm_checkpoint = torch.load(f"{m_config.CKPT_LSTM_DIR}/best_checkpoint.pth", map_location=m_config.DEVICE)

lstm.load_state_dict(lstm_checkpoint["model_state_dict"])
lstm.to("cuda")
lstm.eval()

decoder = old_Decoder(latent_dim=v_config.LATENT_DIM, flatten_dim=v_config.FLATTEN_DIM)
v_checkpoint = torch.load(f'{v_config.CHECKPOINT_DIR}/best_checkpoint.pth', map_location=m_config.DEVICE)
decoder_state_dict = {
    k[len("decoder."):]: v
    for k, v in v_checkpoint["model_state_dict"].items()
    if k.startswith("decoder.")
}
decoder.load_state_dict(decoder_state_dict)
decoder.to("cuda")
decoder.eval()

random_idx = random.randint(0, len(val_dataset) - 1)
z_seq, action_seq, next_z_seq = val_dataset[random_idx]

z_seq = z_seq.unsqueeze(0).to("cuda")         # (1, seq_len, 32)
action_seq = action_seq.unsqueeze(0).to("cuda")  # (1, seq_len, 2)
next_z_seq = next_z_seq.to("cuda")

with torch.no_grad():
    predicted_next_z = lstm(z_seq, action_seq)   # (1, seq_len, 32)
    predicted_next_z = predicted_next_z.squeeze(0)  # (seq_len, 32)

# 5. 시퀀스 중 한 시점만 골라서 이미지로 비교 (예: 마지막 스텝)
step = -1   # 마지막 스텝 기준으로 확인

with torch.no_grad():
    predicted_image = decoder(predicted_next_z[step].unsqueeze(0))   # (1, 3, 96, 96)
    actual_image = decoder(next_z_seq[step].unsqueeze(0))              # (1, 3, 96, 96)

# 6. 시각화
fig, axes = plt.subplots(1, 2, figsize=(6, 3))
axes[0].imshow(actual_image[0].cpu().permute(1, 2, 0))
axes[0].set_title("Actual Next State")
axes[1].imshow(predicted_image[0].cpu().permute(1, 2, 0))
axes[1].set_title("Predicted Next State")
plt.savefig("C:/Users/zxfg0/WorldModel/figures/predicted_u-net_version.png")
plt.show()