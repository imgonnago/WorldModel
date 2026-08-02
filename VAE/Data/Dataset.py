import torch

def ImageDataset(self, Dataset):
        def __init__(self, images):
            self.images = images

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            image = self.images[idx]
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (H, W, C) -> (C, H, W) & normalize
            return image