import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.signal import stft
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

X = np.vstack(X)
snrs = np.array(snrs)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ── 3. Normalize ──────────────────────────────────────────────────────────────
X = X / (np.max(np.abs(X), axis=(1, 2), keepdims=True) + 1e-8)


# ── 4. Convert IQ → Spectrogram ───────────────────────────────────────────────
def iq_to_spectrogram(iq_signal, image_size=64):
    # Combine I and Q into complex signal
    complex_signal = iq_signal[0] + 1j * iq_signal[1]

    # STFT
    _, _, Zxx = stft(complex_signal, nperseg=32, noverlap=24)

    # Magnitude in dB
    spec = np.abs(Zxx)
    spec = 20 * np.log10(spec + 1e-8)

    # Normalize to [0, 255]
    spec -= spec.min()
    spec /= (spec.max() + 1e-8)
    spec = (spec * 255).astype(np.uint8)

    # Resize to image_size x image_size and convert to RGB
    img = Image.fromarray(spec)
    img = img.resize((image_size, image_size), Image.BILINEAR)
    img = img.convert("RGB")
    return np.array(img)


# ── 5. Convert in batches ─────────────────────────────────────────────────────
IMAGE_SIZE = 64
SAVE_PATH = r"C:\Users\Omen\OneDrive\Desktop\Master\Computer_vision\Data\spectrogram_data.npz"
BATCH_SIZE = 10000

print(f"Converting {len(X)} signals to spectrograms at {IMAGE_SIZE}x{IMAGE_SIZE}...")

all_images = []
for batch_start in range(0, len(X), BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, len(X))
    batch = X[batch_start:batch_end]
    batch_images = np.zeros((len(batch), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    for i, sig in enumerate(tqdm(batch,
                                 desc=f"Batch {batch_start // BATCH_SIZE + 1}")):
        batch_images[i] = iq_to_spectrogram(sig, image_size=IMAGE_SIZE)

    all_images.append(batch_images)

X_images = np.concatenate(all_images, axis=0)
print(f"Done! Shape: {X_images.shape}")

# ── 6. Split ──────────────────────────────────────────────────────────────────
X_temp, X_test, y_temp, y_test, snr_temp, snr_test = train_test_split(
    X_images, y_encoded, snrs,
    test_size=0.2, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
    X_temp, y_temp, snr_temp,
    test_size=0.125, random_state=42, stratify=y_temp
)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
np.savez_compressed(
    SAVE_PATH,
    X_train=X_train, X_val=X_val, X_test=X_test,
    y_train=y_train, y_val=y_val, y_test=y_test,
    snr_train=snr_train, snr_val=snr_val, snr_test=snr_test,
    classes=le.classes_
)
print(f"Saved to {SAVE_PATH}")

# ── 8. Preview ────────────────────────────────────────────────────────────────
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
    axes[row, col].set_title(le.classes_[label])
    axes[row, col].axis("off")

plt.suptitle("Spectrogram — One per Modulation", fontsize=14)
plt.tight_layout()
plt.savefig("spectrogram_preview.png", dpi=150)
plt.show()
print("Done!")