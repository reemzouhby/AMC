import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from resnet1d import ResNet1D

# ── 1. MyDataset ──────────────────────────────────────────────────────────────
class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index], dtype=torch.float),
            torch.tensor(self.label[index], dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

# ── 2. Load Data ──────────────────────────────────────────────────────────────
with open(r"C:\Users\Omen\OneDrive\Desktop\Master\Computer_vision\Data\RML2016.10a_dict.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# ── 3. Extract X, y, SNR ──────────────────────────────────────────────────────
X, y, snrs = [], [], []
for (mod, snr), samples in data.items():
    X.append(samples)
    y.extend([mod] * len(samples))
    snrs.extend([snr] * len(samples))

X    = np.vstack(X)
snrs = np.array(snrs)

# ── 4. Encode Labels ──────────────────────────────────────────────────────────
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)
print("X shape:", X.shape)

# ── 5. Normalize ──────────────────────────────────────────────────────────────
X = X / (np.max(np.abs(X), axis=(1, 2), keepdims=True) + 1e-8)

# ── 6. Train / Val / Test Split (70 / 10 / 20) ───────────────────────────────
X_temp, X_test, y_temp, y_test, snr_temp, snr_test = train_test_split(
    X, y_encoded, snrs,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
    X_temp, y_temp, snr_temp,
    test_size=0.125,       # 0.125 x 80% = 10% of total
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape)   # ~154000
print("Val:  ", X_val.shape)     # ~22000
print("Test: ", X_test.shape)    # ~44000

# ── 7. DataLoaders ────────────────────────────────────────────────────────────
train_loader = DataLoader(MyDataset(X_train, y_train), batch_size=128, shuffle=True)
val_loader   = DataLoader(MyDataset(X_val,   y_val),   batch_size=128, shuffle=False)
test_loader  = DataLoader(MyDataset(X_test,  y_test),  batch_size=128, shuffle=False)

# ── 8. Device + Model ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = ResNet1D(
    in_channels=2,
    base_filters=64,
    kernel_size=16,
    stride=2,
    groups=1,
    n_block=8,
    n_classes=11
).to(device)
print("Model parameters:", sum(p.numel() for p in model.parameters()))

# ── 9. Loss + Optimizer ───────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

# ── 10. Training Loop ─────────────────────────────────────────────────────────
EPOCHS = 60
train_losses, val_losses, val_accs = [], [], []
best_val_loss = float("inf")

for epoch in range(EPOCHS):

    # — Train —
    model.train()
    running_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # — Validate —
    model.eval()
    correct, total, running_val_loss = 0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            running_val_loss += criterion(outputs, y_batch).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    avg_val_loss = running_val_loss / len(val_loader)
    val_acc = correct / total * 100
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    scheduler.step(avg_val_loss)

    # — Save best model —
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "resnet1d_best.pth")

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}] "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"Val Acc: {val_acc:.2f}%")

# ── 11. Plot Training Curves ──────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label="Train Loss", color="blue")
ax1.plot(val_losses,   label="Val Loss",   color="orange")
ax1.set_title("Loss Curve - ResNet1D")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True)

ax2.plot(val_accs, label="Val Accuracy", color="green")
ax2.set_title("Accuracy Curve - ResNet1D")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("resnet1d_training_curves.png", dpi=150)
plt.show()
print("Training curves saved!")

# ── 12. Final Test Evaluation ─────────────────────────────────────────────────
model.load_state_dict(torch.load("resnet1d_best.pth"))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        _, predicted = torch.max(model(X_batch), 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)

print(f"\nFinal Test Accuracy: {correct/total*100:.2f}%")