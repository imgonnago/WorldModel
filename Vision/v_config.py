LATENT_DIM = 32
FLATTEN_DIM = 4608
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 500
DEVICE = "cuda"
NUM_EPISODES = 200
MAX_STEPS = 300
CHECKPOINT_DIR = "./checkpoints"
BETA = 0.1
BLUE_WEIGHT = 6.0  # 파란 점(에이전트) 영역에 대한 가중치
PATIENCE = 20  # Early stopping을 위한 patience 값
"""
Epoch 181/500 - Train Loss: 15.6523, Val Loss: 49.8571
  → 베스트 모델 갱신 (val loss: 49.8571)
Epoch 182/500 - Train Loss: 15.6314, Val Loss: 50.0110
Epoch 183/500 - Train Loss: 15.6119, Val Loss: 50.5514
Epoch 184/500 - Train Loss: 15.5924, Val Loss: 50.2791
Epoch 185/500 - Train Loss: 15.5757, Val Loss: 50.0668
Epoch 186/500 - Train Loss: 15.5531, Val Loss: 49.7949
  → 베스트 모델 갱신 (val loss: 49.7949)
Epoch 187/500 - Train Loss: 15.5358, Val Loss: 50.4131
Epoch 188/500 - Train Loss: 15.5236, Val Loss: 50.1489
Epoch 189/500 - Train Loss: 15.5042, Val Loss: 50.0566
Epoch 190/500 - Train Loss: 15.4833, Val Loss: 50.3229
Epoch 191/500 - Train Loss: 15.4639, Val Loss: 49.9526
Epoch 192/500 - Train Loss: 15.4475, Val Loss: 50.2314
Epoch 193/500 - Train Loss: 15.4331, Val Loss: 50.0743
Epoch 194/500 - Train Loss: 15.4148, Val Loss: 49.8160
Epoch 195/500 - Train Loss: 15.3919, Val Loss: 50.8161
Epoch 196/500 - Train Loss: 15.3790, Val Loss: 50.9561
Epoch 197/500 - Train Loss: 15.3646, Val Loss: 50.0277
Epoch 198/500 - Train Loss: 15.3470, Val Loss: 50.4718
Epoch 199/500 - Train Loss: 15.3305, Val Loss: 49.9504
Epoch 200/500 - Train Loss: 15.3142, Val Loss: 49.9841
Epoch 201/500 - Train Loss: 15.2919, Val Loss: 49.8027
Epoch 202/500 - Train Loss: 15.2812, Val Loss: 49.8772
Epoch 203/500 - Train Loss: 15.2682, Val Loss: 50.7883
Epoch 204/500 - Train Loss: 15.2448, Val Loss: 49.9411
Epoch 205/500 - Train Loss: 15.2389, Val Loss: 50.0613
Epoch 206/500 - Train Loss: 15.2143, Val Loss: 49.8211
Early stopping at epoch 206 (patience=20 도달)
Training complete!
"""