import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
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

# ── 2. Load + Preprocess Data ─────────────────────────────────────────────────
with open(r"C:\Users\Omen\OneDrive\Desktop\Master\Computer_vision\Data\RML2016.10a_dict.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

X, y, snrs = [], [], []
for (mod, snr), samples in data.items():
    X.append(samples)
    y.extend([mod] * len(samples))
    snrs.extend([snr] * len(samples))

X    = np.vstack(X)
snrs = np.array(snrs)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X = X / (np.max(np.abs(X), axis=(1, 2), keepdims=True) + 1e-8)

# ── 3. Same Split (random_state=42 keeps it identical) ───────────────────────
X_temp, X_test, y_temp, y_test, snr_temp, snr_test = train_test_split(
    X, y_encoded, snrs, test_size=0.2, random_state=42, stratify=y_encoded)

X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
    X_temp, y_temp, snr_temp, test_size=0.125, random_state=42, stratify=y_temp)

test_loader = DataLoader(MyDataset(X_test, y_test), batch_size=128, shuffle=False)

# ── 4. Load Best Model ────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNet1D(
    in_channels=2, base_filters=64, kernel_size=16,
    stride=2, groups=1, n_block=8, n_classes=11
).to(device)

model.load_state_dict(torch.load("resnet1d_best.pth", weights_only=True))
model.eval()
print("Model loaded! Running evaluation...")

# ── 5. Get Predictions ────────────────────────────────────────────────────────
all_preds = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        _, predicted = torch.max(model(X_batch), 1)
        all_preds.extend(predicted.cpu().numpy())

all_preds = np.array(all_preds)
print(f"Overall Test Accuracy: {accuracy_score(y_test, all_preds)*100:.2f}%")

# ── 6. SNR vs Accuracy ────────────────────────────────────────────────────────
snr_values = sorted(np.unique(snr_test))
snr_accs   = []

for snr in snr_values:
    mask = snr_test == snr
    acc  = accuracy_score(y_test[mask], all_preds[mask]) * 100
    snr_accs.append(acc)
    print(f"SNR {snr:+3d} dB → Accuracy: {acc:.2f}%")

plt.figure(figsize=(12, 5))
plt.plot(snr_values, snr_accs, marker="o", color="blue", linewidth=2)
plt.axhline(y=accuracy_score(y_test, all_preds)*100, color="red",
            linestyle="--", label=f"Overall Accuracy")
plt.title("ResNet1D — Accuracy vs SNR")
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.xticks(snr_values, rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("resnet1d_snr_accuracy.png", dpi=150)
plt.show()

# ── 7. Confusion Matrix ───────────────────────────────────────────────────────
cm = confusion_matrix(y_test, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

plt.figure(figsize=(12, 10))
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("ResNet1D — Confusion Matrix")
plt.tight_layout()
plt.savefig("resnet1d_confusion_matrix.png", dpi=150)
plt.show()

print("All plots saved!")