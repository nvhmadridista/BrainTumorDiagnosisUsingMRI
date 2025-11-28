import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import pandas as pd
import json

from src.models.hybrid_model import BrainTumorModel
from src.dataloader.loader import get_data_loaders
from src.utils.helpers import init_weights

import os

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, acc


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_loss = total_loss / len(val_loader)

    return avg_loss, acc


def main():

    data_dir = "data/processed"
    batch_size = 32
    img_size = 224
    num_epochs = 25
    lr = 1e-4

    MODELS_DIR = '/content/drive/MyDrive/ML/models'
    LOGS_DIR = '/content/drive/MyDrive/ML/logs'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_loader, val_loader, test_loader, class_names = \
        get_data_loaders(data_dir, batch_size, img_size)
    
    model = BrainTumorModel(num_classes=len(class_names)).to(device)
    model.apply(init_weights)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    # ---- Training ----

    history = []
    best_val_loss = float('inf')
    best_val_acc = 0
    patience = 5
    patience_counter = 0

    class_names_path = os.path.join(MODELS_DIR, "class_names.json")
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f)
    print(f"Đã lưu tên lớp: {class_names_path}")

    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        # Lưu mô hình tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0
            save_path = os.path.join(MODELS_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model! Val Loss: {val_loss:.4f}")
        
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Dừng sớm: Val Loss không cải thiện sau {patience} epoch.")
                break

    df_history = pd.DataFrame(history)
    log_path = os.path.join(LOGS_DIR, "training_history.csv")
    df_history.to_csv(log_path, index=False)
    print(f"Đã lưu lịch sử huấn luyện tại: {log_path}")

    print("\nTraining completed!")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()