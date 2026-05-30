import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tqdm import tqdm


# ── Custom Dataset ────────────────────────────────────────────────────────────
class SpectDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data      = data
        self.labels    = labels
        self.transform = transform

    def __getitem__(self, index):
        img = Image.fromarray(self.data[index].astype(np.uint8))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[index], dtype=torch.long)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    print("Loading spectrogram data...")
    npz = np.load(
        r"C:\Users\Omen\OneDrive\Desktop\Master\Computer_vision\Data\spectrogram_data.npz"
    )

    X_train, y_train = npz["X_train"], npz["y_train"]
    X_val,   y_val   = npz["X_val"],   npz["y_val"]
    X_test,  y_test  = npz["X_test"],  npz["y_test"]
    snr_test         = npz["snr_test"]
    classes          = npz["classes"]
    num_classes      = len(classes)

    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"Classes ({num_classes}): {classes}")

    # ── 2. Transforms ─────────────────────────────────────────────────────────
    IMG_SIZE = 112

    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # ── 3. DataLoaders ────────────────────────────────────────────────────────
    BATCH_SIZE  = 128
    NUM_WORKERS = 0

    train_loader = DataLoader(
        SpectDataset(X_train, y_train, transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )
    val_loader = DataLoader(
        SpectDataset(X_val, y_val, transform=eval_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )
    test_loader = DataLoader(
        SpectDataset(X_test, y_test, transform=eval_transform),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0)
    )

    # ── 4. Model ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model_name = "ResNet-18"

    # Freeze early layers — low-level features transfer well from ImageNet
    for name, param in model.named_parameters():
        if any(layer in name for layer in ["layer1", "layer2", "layer3", "conv1", "bn1"]):
            param.requires_grad = False

    # Custom classifier head
    num_features = model.fc.in_features  # 512
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(128, num_classes)
    )

    model = model.to(device)
    print(f"Model: {model_name}")

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    SAVE_PATH = "spectrogram_best.pth"

    # Differential learning rates: deeper layers get slightly higher LR
    optimizer = optim.Adam([
        {"params": model.layer3.parameters(), "lr": 1e-5},
        {"params": model.layer4.parameters(), "lr": 5e-5},
        {"params": model.fc.parameters(), "lr": 5e-4},
    ], weight_decay=3e-3)

    # ── 5. Loss + Scheduler + AMP ─────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    EPOCHS    = 40
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    use_amp = torch.cuda.is_available()
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)
    print(f"AMP (mixed precision): {'enabled ✅' if use_amp else 'disabled (CPU)'}")

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    PATIENCE         = 10
    best_val_loss    = float("inf")
    patience_counter = 0
    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(EPOCHS):

        # — Train —
        model.train()
        running_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1:02d} Train", leave=False):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(imgs), labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # — Validate —
        model.eval()
        correct, total, running_val_loss = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = model(imgs)
                running_val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total   += labels.size(0)

        avg_val_loss = running_val_loss / len(val_loader)
        val_acc      = correct / total * 100
        val_losses.append(avg_val_loss)
        val_accs.append(val_acc)

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss    = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% ✅ saved")
        else:
            patience_counter += 1
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
                  f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% (patience {patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch+1}!")
                break

    # ── 7. Training Curves ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label="Train Loss", color="blue")
    ax1.plot(val_losses,   label="Val Loss",   color="orange")
    ax1.set_title(f"Loss Curve — {model_name} Spectrogram")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.grid(True)

    ax2.plot(val_accs, label="Val Accuracy", color="green")
    ax2.set_title(f"Accuracy Curve — {model_name} Spectrogram")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.savefig("spectrogram_training_curves.png", dpi=150)
    plt.show()
    print("Training curves saved!")

    # ── 8. Final Test Evaluation ──────────────────────────────────────────────
    model.load_state_dict(torch.load(SAVE_PATH, weights_only=True))
    model.eval()

    all_preds = []
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, predicted = torch.max(model(imgs), 1)
            all_preds.extend(predicted.cpu().numpy())

    all_preds   = np.array(all_preds)
    overall_acc = accuracy_score(y_test, all_preds) * 100
    print(f"\nFinal Test Accuracy: {overall_acc:.2f}%")

    # ── 9. SNR vs Accuracy ────────────────────────────────────────────────────
    snr_values = sorted(np.unique(snr_test))
    snr_accs   = []
    for snr in snr_values:
        mask = snr_test == snr
        acc  = accuracy_score(y_test[mask], all_preds[mask]) * 100
        snr_accs.append(acc)
        print(f"SNR {snr:+3d} dB → Accuracy: {acc:.2f}%")

    plt.figure(figsize=(12, 5))
    plt.plot(snr_values, snr_accs, marker="o", color="darkorange", linewidth=2)
    plt.axhline(y=overall_acc, color="blue", linestyle="--",
                label=f"Overall Accuracy ({overall_acc:.2f}%)")
    plt.title(f"{model_name} Spectrogram — Accuracy vs SNR")
    plt.xlabel("SNR (dB)"); plt.ylabel("Accuracy (%)")
    plt.xticks(snr_values, rotation=45)
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("spectrogram_snr_accuracy.png", dpi=150)
    plt.show()

    # ── 10. Confusion Matrix ──────────────────────────────────────────────────
    cm   = confusion_matrix(y_test, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap="Oranges", xticks_rotation=45, ax=ax)
    plt.title(f"{model_name} Spectrogram — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("spectrogram_confusion_matrix.png", dpi=150)
    plt.show()

    print("All plots saved!")