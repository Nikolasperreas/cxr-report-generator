"""Encoders: CNNs that turn an X-ray into a feature vector, plus a classification head."""

from torch import nn


def conv_block(in_channels, out_channels):
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2: channels in -> out, height and width halved."""
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
    batch_norm = nn.BatchNorm2d(out_channels)
    relu = nn.ReLU()
    max_pool = nn.MaxPool2d(2)
    return nn.Sequential(conv, batch_norm, relu, max_pool)


class SmallCNN(nn.Module):
    """Small CNN from scratch: features (N, channels[-1]) from conv blocks + global average
    pooling, and a linear head giving logits (N, n_classes)."""

    def __init__(self, channels, n_classes):
        super().__init__()
        # the smallcnn class is a kind on nn.Module(superclass)
        # 1. blocks: one conv_block per entry of channels; the first takes 1 input channel
        blocks = []
        blocks.append(conv_block(1, channels[0]))
        for i in range(0, len(channels) - 1):
            blocks.append(conv_block(channels[i], channels[i + 1]))
        #    (grayscale), each next one takes the previous block's out_channels
        # 2. self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.features = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())

        # 3. self.head = nn.Linear(channels[-1], n_classes)
        self.head = nn.Linear(channels[-1], n_classes)

    # model = SmallCNN([16, 32, 64, 128, 256], 14)    # __init__ runs: the layers are built
    # images, labels = next(iter(loader))              # images: (16, 1, 224, 224)
    # logits = model(images)                           # ← this line calls forward(self, x) with x = images
    def forward(self, x):
        # features, then head -> logits (N, n_classes)
        features = self.features(x)
        logits = self.head(features)

        return logits

    # x                      (16,   1, 224, 224)    16 X-rays
    # → block 1              (16,  16, 112, 112)
    # → block 2              (16,  32,  56,  56)
    # → block 3              (16,  64,  28,  28)
    # → block 4              (16, 128,  14,  14)
    # → block 5              (16, 256,   7,   7)
    # → avg pool             (16, 256,   1,   1)
    # → flatten              (16, 256)              ← features: 256 numbers per X-ray
