import m_config
from torch.utils.data import DataLoader
from m_model.LSTM import LSTM
from train import train_lstm
from SequenceDataset import get_datasets
from torchinfo import summary
import random
import matplotlib.pyplot as plt


print("=========LSTM train run=========")
print('\n')
print("get dataset")
print('\n')

train_dataset, val_dataset = get_datasets()

train_loader = DataLoader(train_dataset, batch_size=m_config.TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=m_config.TRAIN_BATCH_SIZE, shuffle=False)

print(f'data load complete.')
print(f'train: {len(train_loader)}, val: {len(val_loader)}')
print('\n')

print("model load")

lstm, trained_loss, val_loss = LSTM(action_dim=m_config.ACTION_DIM, hidden_size=m_config.HIDDEN_SIZE)
summary(lstm)

print("model LSTM loaded")
print("\n")

print("model train...")

lstm = train_lstm(
    lstm,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=m_config.EPOCHS,
    lr=m_config.LR,
    device=m_config.DEVICE,
    checkpoint_dir=m_config.CKPT_LSTM_DIR,
    patience=m_config.PATIENCE,
)

print("model train complete!")

random_num = random.randint(0,500)
plt.plot(trained_loss, label='train loss')
plt.plot(val_loss, label='val loss')
plt.title("Model Fitting loss")
plt.xlabel("Epoch")
plt.ylabel("loss") 
plt.legend()
plt.savefig(f"C:/Users/zxfg0/WorldModel/figures/LSTMModelLoss{random_num}.png")
plt.show   

