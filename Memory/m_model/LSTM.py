import torch
import torch.nn as nn
import Memory.m_config as m_config
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

class LSTM(nn.Module):
    def __init__(self, latent_dim=m_config.LATENT_DIM, action_dim=m_config.ACTION_DIM, hidden_size=m_config.HIDDEN_SIZE ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=latent_dim + action_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc_out = nn.Linear(hidden_size, latent_dim)

    def forward(self, z, action):
        x = torch.cat([z , action], dim=-1)
        output, (h_n, c_n) = self.lstm(x) 
        predicted_next_z = self.fc_out(output)  
        return predicted_next_z

