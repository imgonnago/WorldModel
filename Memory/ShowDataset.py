from torch.utils.data import Dataset
import torch
import Vision.v_config as v_config
import Memory.m_config as m_config
import numpy as np

class ShowDataset(Dataset):
    def __init__(self, obs, action, next_obs, episode_len=300, seq_len=30):
        self.obs = np.load(obs)
        self.action = np.load(action)
        self.next_obs = np.load(next_obs)
        self.episode_len = episode_len
        self.seq_len = seq_len

        num_episode = len(self.obs) // self.episode_len
        self.start_indices = []
        for ep in range(num_episode):
            ep_start = ep * episode_len
            for i in range(0, episode_len, seq_len):
                self.start_indices.append(ep_start + i)

    def __len__(self):
        return len(self.start_indices)

    def _to_tensor_image(self,img):
        img = torch.tensor(img, dtype=torch.float32) / 255.0
        return img.permute(2,0,1)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.seq_len

        obs_seq = self.obs[start:end]
        obs_seq = torch.stack([self._to_tensor_image(o) for o in obs_seq])

        action_seq = torch.tensor(self.action[start:end], dtype=torch.float32)

        next_obs_seq = self.next_obs[start:end]
        next_obs_seq = torch.stack([self._to_tensor_image(o) for o in next_obs_seq])

        return obs_seq, action_seq, next_obs_seq


def get_show_datasets(episode_len=300, seq_len=m_config.SEQUENCE_LEN):
    train_dataset = ShowDataset(
        obs=m_config.OBS_TRAIN_DIR,
        action=m_config.ACTION_TRAIN_DIR,
        next_obs=m_config.NEXT_OBS_TRAIN_DIR,
        episode_len=episode_len,
        seq_len=seq_len
    )

    val_dataset = ShowDataset(
        obs=m_config.OBS_VAL_DIR,
        action=m_config.ACTION_VAL_DIR,
        next_obs=m_config.NEXT_OBS_VAL_DIR,
        episode_len=episode_len,
        seq_len=seq_len
    )

    return train_dataset, val_dataset