# RF Modulation Classification

A deep learning project for automatic modulation classification (AMC) on the **RML2016.10a** dataset, comparing multiple model architectures across 1D (raw IQ) and 2D (image-based) input representations.

---

## Dataset

**RadioML 2016.10a** — a standard benchmark for AMC containing 220,000 labeled radio signal samples across 11 modulation types and 20 SNR levels (−20 dB to +18 dB).

**Modulation classes:** 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM

**Split:** 70% train / 10% validation / 20% test (stratified, `random_state=42`)

---

## Project Structure

```
├── 1D/                                    # Raw IQ signal models
│   ├── resnet1d.py                        # ResNet1D architecture
│   ├── googlenet1d.py                     # GoogLeNet1D architecture
│   ├── Model1.py                          # Train ResNet1D
│   ├── Model2.py                          # Train GoogLeNet1D
│   ├── evaluate.py                        # Evaluate ResNet1D
│   ├── evaluategooglenet.py               # Evaluate GoogLeNet1D
│   
│   ├── resnet1d_training_curves.png
│   ├── resnet1d_snr_accuracy.png
│   ├── resnet1d_confusion_matrix.png
│   ├── googlenet1d_training_curves.png
│   ├── googlenet1d_snr_accuracy.png
│   └── googlenet1d_confusion_matrix.png
│
├── 2D/                                    # Image-based models (transfer learning)
│   ├── STFT.py                            # IQ → STFT spectrogram (64×64 RGB)
│   ├── convert_constellation.py           # IQ → constellation diagram (64×64 RGB)
│   ├── ModelResnet.py                     # Train ResNet18 on constellation images
│   ├── Modelspect.py                      # Train ResNet18 on spectrogram images
│   ├── googlenet2d_constellation.py       # Train GoogLeNet on constellation images
│   ├── googlenet2d_spectrogram.py         # Train GoogLeNet on spectrogram images
│   
│   ├── constellation_preview.png
│   ├── resnet2d_constellation_training_curves.png
│   ├── resnet2d_constellation_snr_accuracy.png
│   ├── resnet2d_constellation_confusion_matrix.png
│   ├── spectrogram_training_curves.png
│   ├── spectrogram_snr_accuracy.png
│   ├── spectrogram_confusion_matrix.png
│   ├── googlenet2d_constellation_training_curves.png
│   ├── googlenet2d_constellation_snr_accuracy.png
│   ├── googlenet2d_constellation_confusion_matrix.png
│   ├── googlenet2d_spectrogram_training_curves.png
│   ├── googlenet2d_spectrogram_snr_accuracy.png
│   └── googlenet2d_spectrogram_confusion_matrix.png
│
├── Automatic Modulation Classification Using...pptx   # Project presentation
└── Note_dataset.txt                                   # Dataset notes
```

---

## Models

### 1D Models (Raw IQ Input)

| Model | Input | Architecture |
|---|---|---|
| **ResNet1D** | (2, 128) IQ signal | 8-block 1D residual network, base filters=64, kernel=16 |
| **GoogLeNet1D** | (2, 128) IQ signal | 1D Inception network with 9 Inception blocks + GAP |

Both models take raw normalized IQ samples directly — no feature engineering required.

### 2D Models (Image Input — Transfer Learning)

| Model | Input Representation | Backbone | Image Size |
|---|---|---|---|
| **ResNet18 Constellation** | Constellation diagram | ResNet-18 (ImageNet) | 96×96 |
| **ResNet18 Spectrogram** | STFT spectrogram | ResNet-18 (ImageNet) | 112×112 |
| **GoogLeNet Constellation** | Constellation diagram | GoogLeNet (ImageNet) | 96×96 |
| **GoogLeNet Spectrogram** | STFT spectrogram | GoogLeNet (ImageNet) | 112×112 |

All 2D models use pretrained ImageNet weights with early layers frozen and a custom classification head:
`Linear(→256) → BN → ReLU → Dropout → Linear(→128) → BN → ReLU → Dropout → Linear(→11)`

---

## Input Representations

### Raw IQ (1D)
Each sample is a `(2, 128)` array of normalized I and Q channel values, fed directly to 1D convolutional networks.

### Constellation Diagrams (2D)
I/Q samples are plotted as a scatter diagram in the complex plane and saved as a 64×64 RGB image. The resulting images capture the geometric structure of each modulation scheme.

![Constellation Preview](constellation_preview.png)

### STFT Spectrograms (2D)
IQ signals are converted to complex form and processed with Short-Time Fourier Transform (`nperseg=32, noverlap=24`). Magnitude is computed in dB, normalized to `[0, 255]`, and saved as a 64×64 RGB image.

---

## Training Configuration

| Setting | 1D Models | 2D Models |
|---|---|---|
| Optimizer | Adam | Adam (differential LRs) |
| Loss | CrossEntropyLoss | CrossEntropyLoss (label smoothing=0.1) |
| Scheduler | ReduceLROnPlateau | CosineAnnealingLR |
| Batch size | 128 | 128 |
| Max epochs | 60 | 40 |
| Early stopping | 10 epochs patience | 10 epochs patience |
| Gradient clipping | — | max_norm=1.0 |
| Mixed precision | — | AMP (when CUDA available) |

**Data augmentation (2D constellation only):** random horizontal/vertical flip, random rotation ±180°, color jitter — exploiting the rotational symmetry of IQ constellation diagrams.

---

## Results

### Overall Test Accuracy

| Model | Accuracy |
|---|---|
| ResNet1D | ~62% |
| GoogLeNet1D | 61.18% |
| ResNet18 Constellation | — |
| ResNet18 Spectrogram | — |
| GoogLeNet Constellation | — |
| GoogLeNet Spectrogram | — |

### Accuracy vs SNR

Both 1D models show the characteristic S-curve behavior of AMC systems:
- **Low SNR (≤ −10 dB):** ~10% accuracy (near-random, 11 classes)
- **Transition zone (−10 to 0 dB):** rapid improvement
- **High SNR (≥ +2 dB):** ~90% accuracy, plateauing

---

## Key Observations

- **AM-SSB confusion:** Both 1D models heavily misclassify other modulations as AM-SSB. This is a known challenge — AM-SSB occupies only one sideband and its constellation is ambiguous at low SNR.
- **QAM16/QAM64 confusion:** Higher-order QAM schemes are frequently confused with each other, especially at mid-range SNR where constellation points blur together.
- **WBFM confusion:** WBFM is often misclassified as AM-DSB at low SNR due to overlapping spectral characteristics.
- **High-SNR ceiling:** All models approach ~90–91% at high SNR, suggesting a shared performance ceiling likely imposed by the dataset itself rather than model capacity.

---

## Installation

```bash
pip install torch torchvision numpy scikit-learn matplotlib scipy tqdm Pillow
```

**Python:** 3.8+  
**PyTorch:** 1.12+ (CUDA recommended for 2D models)

---

## Usage

### 1. Prepare Data (2D only)

```bash
cd 2D

# Generate STFT spectrogram images
python STFT.py

# Generate constellation diagram images
python convert_constellation.py
```

### 2. Train

```bash
# 1D models (raw IQ)
cd 1D
python Model1.py          # ResNet1D
python Model2.py          # GoogLeNet1D

# 2D models (images)
cd 2D
python ModelResnet.py                  # ResNet18 + constellation
python Modelspect.py                   # ResNet18 + spectrogram
python googlenet2d_constellation.py   # GoogLeNet + constellation
python googlenet2d_spectrogram.py     # GoogLeNet + spectrogram
```

### 3. Evaluate

```bash
cd 1D
python evaluate.py          # ResNet1D — SNR curve + confusion matrix
python evaluategooglenet.py # GoogLeNet1D — SNR curve + confusion matrix
```

Each evaluation script saves:
- `*_snr_accuracy.png` — per-SNR accuracy curve
- `*_confusion_matrix.png` — full 11-class confusion matrix

---

## References

- T. J. O'Shea and J. Hoydis, "An Introduction to Deep Learning for the Physical Layer," *IEEE TCCN*, 2017.
- RML2016.10a dataset: [https://www.deepsig.ai/datasets](https://www.deepsig.ai/datasets)
- He et al., "Deep Residual Learning for Image Recognition," *CVPR*, 2016.
- Szegedy et al., "Going Deeper with Convolutions," *CVPR*, 2015.
