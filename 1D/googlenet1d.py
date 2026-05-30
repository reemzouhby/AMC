import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 1. Basic 1D Conv Block ────────────────────────────────────────────────────
class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn   = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ── 2. Inception Block (1D) ───────────────────────────────────────────────────
class InceptionBlock1D(nn.Module):
    """
    4 parallel branches:
        Branch 1: 1x1 conv
        Branch 2: 1x1 conv → 1x3 conv
        Branch 3: 1x1 conv → 1x5 conv
        Branch 4: MaxPool  → 1x1 conv
    """
    def __init__(self, in_channels, out_1x1, out_3x3_reduce, out_3x3,
                 out_5x5_reduce, out_5x5, out_pool):
        super(InceptionBlock1D, self).__init__()

        # Branch 1: 1x1
        self.branch1 = Conv1dBlock(in_channels, out_1x1, kernel_size=1)

        # Branch 2: 1x1 → 3x1
        self.branch2 = nn.Sequential(
            Conv1dBlock(in_channels, out_3x3_reduce, kernel_size=1),
            Conv1dBlock(out_3x3_reduce, out_3x3, kernel_size=3, padding=1)
        )

        # Branch 3: 1x1 → 5x1
        self.branch3 = nn.Sequential(
            Conv1dBlock(in_channels, out_5x5_reduce, kernel_size=1),
            Conv1dBlock(out_5x5_reduce, out_5x5, kernel_size=5, padding=2)
        )

        # Branch 4: MaxPool → 1x1
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            Conv1dBlock(in_channels, out_pool, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # concat on channel dim


# ── 3. GoogLeNet1D ────────────────────────────────────────────────────────────
class GoogLeNet1D(nn.Module):
    """
    Input:  (batch, 2, 128)   — I/Q signal
    Output: (batch, n_classes)

    Architecture mirrors GoogLeNet structure adapted for 1D RF signals
    """
    def __init__(self, in_channels=2, n_classes=11):
        super(GoogLeNet1D, self).__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            Conv1dBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            Conv1dBlock(64, 64,  kernel_size=1),
            Conv1dBlock(64, 192, kernel_size=3, padding=1),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # ── Inception Blocks ──────────────────────────────────────────────────
        # inception_3a: output channels = 64+128+32+32 = 256
        self.inception_3a = InceptionBlock1D(192, 64, 96,  128, 16, 32, 32)

        # inception_3b: output channels = 128+192+96+64 = 480
        self.inception_3b = InceptionBlock1D(256, 128, 128, 192, 32, 96, 64)

        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # inception_4a: output channels = 192+208+48+64 = 512
        self.inception_4a = InceptionBlock1D(480, 192, 96,  208, 16, 48, 64)

        # inception_4b: output channels = 160+224+64+64 = 512
        self.inception_4b = InceptionBlock1D(512, 160, 112, 224, 24, 64, 64)

        # inception_4c: output channels = 128+256+64+64 = 512
        self.inception_4c = InceptionBlock1D(512, 128, 128, 256, 24, 64, 64)

        # inception_4d: output channels = 112+288+64+64 = 528
        self.inception_4d = InceptionBlock1D(512, 112, 144, 288, 32, 64, 64)

        # inception_4e: output channels = 256+320+128+128 = 832
        self.inception_4e = InceptionBlock1D(528, 256, 160, 320, 32, 128, 128)

        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # inception_5a: output channels = 256+320+128+128 = 832
        self.inception_5a = InceptionBlock1D(832, 256, 160, 320, 32, 128, 128)

        # inception_5b: output channels = 384+384+128+128 = 1024
        self.inception_5b = InceptionBlock1D(832, 384, 192, 384, 48, 128, 128)

        # ── Classifier ────────────────────────────────────────────────────────
        self.gap      = nn.AdaptiveAvgPool1d(1)   # global average pooling
        self.dropout  = nn.Dropout(p=0.5)
        self.fc       = nn.Linear(1024, n_classes)

    def forward(self, x):
        # Stem
        out = self.stem(x)

        # Inception 3
        out = self.inception_3a(out)
        out = self.inception_3b(out)
        out = self.pool3(out)

        # Inception 4
        out = self.inception_4a(out)
        out = self.inception_4b(out)
        out = self.inception_4c(out)
        out = self.inception_4d(out)
        out = self.inception_4e(out)
        out = self.pool4(out)

        # Inception 5
        out = self.inception_5a(out)
        out = self.inception_5b(out)

        # Classify
        out = self.gap(out)
        out = out.view(out.size(0), -1)   # flatten
        out = self.dropout(out)
        out = self.fc(out)

        return out