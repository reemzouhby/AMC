import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import io
import os
from tqdm import tqdm

# ── 1. Load Data ──────────────────────────────────────────────────────────────
print("Loading data...")
with open(r"C:\Users\Omen\OneDrive\Desktop\Master\Computer_vision\Data\RML2016.10a_dict.pkl", "rb") as f:
    data = pickle.load(f, encoding="latin1")

# ── 2. Extract X, y, SNR ──────────────────────────────────────────────────────
X, y, snrs = [], [], []
for (mod, snr), samples in data.items():
    X.append(samples)
    y.extend([mod] * len(samples))
    snrs.extend([snr] * len(samples))

X    = np.vstack(X)
snrs = np.array(snrs)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ── 3. Normalize ──────────────────────────────────────────────────────────────
X = X / (np.max(np.abs(X), axis=(1, 2), keepdims=True) + 1e-8)

# ── 4. Convert function ───────────────────────────────────────────────────────
def iq_to_constellation(iq_signal, image_size=64):
    I = iq_signal[0]
    Q = iq_signal[1]

    fig, ax = plt.subplots(figsize=(1, 1), dpi=image_size)
    ax.scatter(I, Q, s=3, c="blue", alpha=0.6)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)

# ── 5. Convert in batches + save directly ─────────────────────────────────────
IMAGE_SIZE  = 64    # 64x64 instead of 224x224 → 30.8GB → 1.7GB ✅
SAVE_PATH   = r"C:\Users\Omen\OneDrive\Desktop\Master\Computer_vision\Data\constellation_data.npz"
BATCH_SIZE  = 10000

print(f"Converting {len(X)} signals at {IMAGE_SIZE}x{IMAGE_SIZE}...")
print(f"Memory needed: {len(X) * IMAGE_SIZE * IMAGE_SIZE * 3 / 1e9:.2f} GB")

# Convert all in batches to avoid memory spike
all_images = []

for batch_start in range(0, len(X), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(X))
    batch     = X[batch_start:batch_end]

    batch_images = np.zeros((len(batch), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    for i, sig in enumerate(tqdm(batch, desc=f"Batch {batch_start//BATCH_SIZE + 1}/{len(X)//BATCH_SIZE + 1}")):
        batch_images[i] = iq_to_constellation(sig, image_size=IMAGE_SIZE)

    all_images.append(batch_images)
    print(f"  Batch done: {batch_start} → {batch_end}")

X_images = np.concatenate(all_images, axis=0)
print(f"Done! Shape: {X_images.shape}")

# ── 6. Split ──────────────────────────────────────────────────────────────────
print("Splitting...")
X_temp, X_test, y_temp, y_test, snr_temp, snr_test = train_test_split(
    X_images, y_encoded, snrs,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)
X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
    X_temp, y_temp, snr_temp,
    test_size=0.125,
    random_state=42,
    stratify=y_temp
)

print("Train:", X_train.shape)
print("Val:  ", X_val.shape)
print("Test: ", X_test.shape)

# ── 7. Save ───────────────────────────────────────────────────────────────────
print("Saving...")
np.savez_compressed(
    SAVE_PATH,
    X_train=X_train, X_val=X_val,       X_test=X_test,
    y_train=y_train, y_val=y_val,        y_test=y_test,
    snr_train=snr_train, snr_val=snr_val, snr_test=snr_test,
    classes=le.classes_
)
print(f"Saved to {SAVE_PATH}")

# ── 8. Preview ────────────────────────────────────────────────────────────────
print("Saving preview...")
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
shown = {}
for i in range(len(X_images)):
    label = y_encoded[i]
    if label not in shown:
        shown[label] = X_images[i]
    if len(shown) == 10:
        break

for i, (label, img) in enumerate(shown.items()):
    row, col = i // 5, i % 5
    axes[row, col].imshow(img)
    axes[row, col].set_title(le.classes_[label], fontsize=10)
    axes[row, col].axis("off")

plt.suptitle("Constellation Diagrams — One per Modulation", fontsize=14)
plt.tight_layout()
plt.savefig("constellation_preview.png", dpi=150)
plt.show()
print("All done!")