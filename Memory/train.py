import torch
import torch.nn.functional as F
import os
import m_config


def m_loss(predicted_next_z, next_z):
    return F.mse_loss(predicted_next_z, next_z, reduction='sum')


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


def train_lstm(
    model,
    train_loader,
    val_loader,
    epochs=m_config.EPOCHS,
    lr=m_config.LR,
    device='cuda',
    checkpoint_dir=m_config.CKPT_LSTM_DIR,
    patience=m_config.PATIENCE,
    resume_from=None,
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start_epoch = 0
    best_val_loss = float("inf")
    no_improve_count = 0
    train_loss_history = []
    val_loss_history = []

    if resume_from is not None and os.path.exists(resume_from):
        model, optimizer, start_epoch = load_checkpoint(model, optimizer, resume_from, device)

    for epoch in range(start_epoch, epochs):
        # --- Train ---
        model.train()
        train_loss = 0
        for z_batch, action_batch, next_z_batch in train_loader:
            z_batch = z_batch.to(device)
            action_batch = action_batch.to(device)
            next_z_batch = next_z_batch.to(device)

            optimizer.zero_grad()
            predicted_next_z = model(z_batch, action_batch)
            loss = m_loss(predicted_next_z, next_z_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_loss_history.append(avg_train_loss)
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for z_batch, action_batch, next_z_batch in val_loader:
                z_batch = z_batch.to(device)
                action_batch = action_batch.to(device)
                next_z_batch = next_z_batch.to(device)

                predicted_next_z = model(z_batch, action_batch)
                loss = m_loss(predicted_next_z, next_z_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_loss_history.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        save_checkpoint(
            model, optimizer, epoch, avg_train_loss, avg_val_loss,
            save_path=os.path.join(checkpoint_dir, "last_checkpoint.pth")
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_count = 0
            save_checkpoint(
                model, optimizer, epoch, avg_train_loss, avg_val_loss,
                save_path=os.path.join(checkpoint_dir, "best_checkpoint.pth")
            )
            print(f"  → 베스트 모델 갱신 (val loss: {avg_val_loss:.4f})")
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                print(f"Early stopping at epoch {epoch+1} (patience={patience} 도달)")
                break

    return model, train_loss_history, val_loss_history