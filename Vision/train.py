import torch
import numpy as np
import Vision.v_config as v_config
import torch.nn as nn
import torch.nn.functional as F
import os

def vae_loss(reconstructed, original, mu, logvar, beta=v_config.BETA, blue_weight=v_config.BLUE_WEIGHT):
    # 파란 점(에이전트) 영역에 대한 가중치를 적용한 MSE 손실 계산
    blue_mask = (
        (original[:, 2] > 0.85) &
        (original[:, 0] < 0.40) &
        (original[:, 1] > 0.35) & (original[:, 1] < 0.60)
    )
    blue_mask = blue_mask.unsqueeze(1).float()

    weight_map = torch.ones_like(original)
    weight_map = weight_map + blue_mask * (blue_weight - 1)
    # 원본 이미지와 재구성 이미지 간의 MSE 손실 계산
    recon_loss = ((reconstructed - original) ** 2 * weight_map).sum()
    # KL divergence는 평균과 분산을 사용하여 계산
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, save_path):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, save_path)


def load_checkpoint(model, optimizer, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"체크포인트 로드 완료: epoch {checkpoint['epoch']}부터 이어서 학습")
    return model, optimizer, start_epoch


def train_vae(
    model,
    train_loader,
    val_loader,
    epochs=20,
    lr=1e-4,
    device="cuda",
    checkpoint_dir=v_config.CHECKPOINT_DIR,
    resume_from=None,
    beta=v_config.BETA,
    patience=v_config.PATIENCE
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    no_improvement_epochs = 0
    best_val_loss = float("inf")

    if resume_from is not None and os.path.exists(resume_from):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, resume_from, device)

    for epoch in range(start_epoch, epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            reconstructed, mu, logvar = model(batch)
            loss = vae_loss(reconstructed, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)

        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed, mu, logvar = model(batch)
                loss = vae_loss(reconstructed, batch, mu, logvar)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- 매 epoch마다 최신 체크포인트 저장 (재개용) ---
        save_checkpoint(
            model, optimizer, epoch, avg_train_loss, avg_val_loss,
            save_path=os.path.join(checkpoint_dir, "last_checkpoint.pth")
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            save_checkpoint(model, optimizer, epoch, avg_train_loss, avg_val_loss, save_path=os.path.join(checkpoint_dir, "best_checkpoint.pth"))
            print(f"  → 베스트 모델 갱신 (val loss: {avg_val_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience} 도달)")
                break

    return model