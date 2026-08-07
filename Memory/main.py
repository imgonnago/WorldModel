import m_config
from torch.utils.data import DataLoader
from m_model.LSTM import LSTM
from train import train_lstm
from SequenceDataset import get_datasets


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

lstm = LSTM(action_dim=m_config.ACTION_DIM, hidden_size=m_config.HIDDEN_SIZE)

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

