import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
LATENT_DIM = 32
FLATTEN_DIM = 4608
BATCH_SIZE = 64
LR = 1e-4
EPOCHS = 200
DEVICE = "cuda"
NUM_EPISODES = 200
MAX_STEPS = 300
CHECKPOINT_DIR = "./checkpoints"
BETA = 0.1
BLUE_WEIGHT = 6.0  # 파란 점(에이전트) 영역에 대한 가중치
PATIENCE = 20  # Early stopping을 위한 patience 값