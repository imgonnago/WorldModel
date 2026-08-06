import torch
from torch.utils.data import Dataset
import numpy as np
import m_config

class SequenceDataset(Dataset):
    def __init__(self, z, actions, next_z, episode_len=300, seq_len=30):
        self.z = np.load(z)
        self.actions = np.load(actions)   
        self.next_z = np.load(next_z)    
        self.episode_len = episode_len
        self.seq_len = seq_len

        num_episode = len(self.z) // self.episode_len
        self.start_indices = []
        for ep in range(num_episode):
            ep_start = ep * episode_len
            for i in range(0, episode_len, seq_len):
                self.start_indices.append(ep_start + i)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start = self.start_indices[idx]
        end = start + self.seq_len

        z_seq = self.z[start:end]         
        action_seq = self.actions[start:end] 
        next_z_seq = self.next_z[start:end] 

        return (
            torch.tensor(z_seq, dtype=torch.float32),
            torch.tensor(action_seq, dtype=torch.float32),
            torch.tensor(next_z_seq, dtype=torch.float32),
        )

    def get_datasets(episode_len=300, seq_len=m_config.SEQUENCE_LEN):
        train_dataset = SequenceDataset(
            z=m_config.Z_TRAIN_DIR,
            actions=m_config.ACTION_TRAIN_DIR,
            next_z=m_config.NEXT_Z_TRAIN_DIR,
            episode_len=episode_len,
            seq_len=seq_len,
        )
        val_dataset = SequenceDataset(
            z=m_config.Z_VAL_DIR,
            actions=m_config.ACTION_VAL_DIR,
            next_z=m_config.NEXT_Z_VAL_DIR,
            episode_len=episode_len,
            seq_len=seq_len,
        )
        return train_dataset, val_dataset