# project/common/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Torchvision for pretrained backbones
    from torchvision import models

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class SimpleCNN(nn.Module):
    """
    מודל קטן ופשוט לבדיקות מהירות / ריצות דיבאג.
    מתאים לתמונות 128x128 או 224x224, 1 או 3 ערוצים.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /8
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # יוציא (B, 128, 1, 1)
            nn.Flatten(),  # (B, 128)
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


def _make_resnet18(
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """
    ResNet-18 פרה-טריינד כ–backbone ל–Chest X-Ray.
    """

    if not HAS_TORCHVISION:
        raise ImportError(
            "torchvision is required for ResNet-18 backbone. "
            "Install it or use SimpleCNN via get_model(arch='simple')."
        )

    # תמיכה ב–API החדש/ישן של torchvision
    try:
        from torchvision.models import resnet18, ResNet18_Weights

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
    except Exception:
        from torchvision.models import resnet18

        model = resnet18(pretrained=pretrained)

    # התאמת מספר ערוצים (למשל 1 ערוץ ל–X-ray אפור)
    if in_channels != 3:
        old_conv = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

    # החלפת השכבה האחרונה ל–num_classes
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def get_model(
    arch: str = "resnet18",
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = True,
) -> nn.Module:
    """
    מפעל מודלים מאוחד:
      - arch='simple'   -> SimpleCNN
      - arch='resnet18' -> ResNet-18 פרה-טריינד
    """

    arch = arch.lower()

    if arch == "simple":
        return SimpleCNN(num_classes=num_classes, in_channels=in_channels)

    if arch in ("resnet", "resnet18"):
        return _make_resnet18(
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=pretrained,
        )

    raise ValueError(f"Unknown architecture: {arch}")
