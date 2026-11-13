"""Encoders: CNNs that turn an X-ray into a feature vector, plus a classification head."""

import torch
from torch import nn
from torchvision.models import resnet50, mobilenet_v2

# ImageNet statistics per channel (R, G, B): the scale the pretrained weights were trained on
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
        for i in range(len(channels) - 1):
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


class PretrainedClassifier(nn.Module):
    """An ImageNet-pretrained backbone + a new linear head giving logits (N, n_classes).
    Takes our grayscale images (N, 1, H, W) in 0..1, like SmallCNN. Parent class of
    ResNet50Classifier and MobileNetV2Classifier, which only build their backbone."""

    def __init__(self, backbone, n_features, n_classes):
        super().__init__()
        # 1. the backbone, already without its ImageNet head: returns (N, n_features)
        self.features = backbone
        # 2. our head: n_features -> n_classes logits
        self.head = nn.Linear(n_features, n_classes)
        # 3. mean and std as buffers: they move with .to(device) and are saved, never trained;
        #    shape (1, 3, 1, 1) so they line up with the channel axis of (N, 3, H, W)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.expand(
            -1, 3, -1, -1
        )  # gray (N, 1, H, W) -> 3 identical channels (N, 3, H, W)
        x = (x - self.mean) / self.std  # the ImageNet scale the weights expect
        features = self.features(x)
        logits = self.head(features)
        return logits

    def freeze_backbone(self):
        """Stage 1: no gradients for the backbone, so only the head trains."""
        self.features.requires_grad_(False)

    def unfreeze(self, names):
        """Stage 2: these parts of the backbone train too, by name, dotted when nested:
        ["layer4"] (ResNet50), ["features.18"] (MobileNetV2)."""
        for name in names:
            self.features.get_submodule(name).requires_grad_(True)

    def train(self, mode=True):
        """As nn.Module.train, but the backbone's BatchNorm layers always stay in eval mode:
        they keep normalizing with ImageNet's stored statistics and never change them
        (requires_grad=False does not stop that; train mode does)."""
        super().train(
            mode
        )  # the usual: every layer to train mode (True) or eval mode (False)
        for layer in self.features.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self  # nn.Module.train returns the model; keep that behaviour


class ResNet50Classifier(PretrainedClassifier):
    """ResNet50 pretrained on ImageNet: features (N, 2048)."""

    def __init__(self, n_classes, weights):
        # the backbone goes in a local variable first: nothing may be put on self
        # before nn.Module.__init__ (inside super().__init__) has run
        # weights: "IMAGENET1K_V1" from the config; None = random
        backbone = resnet50(weights=weights)
        # the ImageNet head is fc = Linear(2048, 1000): read its input size BEFORE removing it
        n_features = backbone.fc.in_features
        # remove the ImageNet head: fc now returns its input, the 2048 averaged features
        backbone.fc = nn.Identity()
        super().__init__(backbone, n_features, n_classes)


class MobileNetV2Classifier(PretrainedClassifier):
    """MobileNetV2 pretrained on ImageNet: features (N, 1280). Built for phones:
    depthwise-separable convolutions, ~10x fewer parameters than ResNet50."""

    def __init__(self, n_classes, weights):
        backbone = mobilenet_v2(weights=weights)
        # torchvision's ImageNet head is `classifier` = Dropout(0.2) -> Linear(1280, 1000):
        # read the Linear's input size, then replace the head whole, so no dropout
        # (like our ResNet head)
        n_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        super().__init__(backbone, n_features, n_classes)


# the pretrained encoders a config can name (`encoder: resnet50`), for the training script
PRETRAINED_ENCODERS = {
    "resnet50": ResNet50Classifier,
    "mobilenet_v2": MobileNetV2Classifier,
}
