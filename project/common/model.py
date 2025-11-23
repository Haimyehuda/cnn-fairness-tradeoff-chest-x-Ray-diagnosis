# model.py
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """Small CNN for 32x32 images (CIFAR-10 cats & dogs)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class BiggerCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # אם עושים pool אחרי conv1, conv2, conv3:
        # 32x32 -> 16x16 -> 8x8 -> 4x4
        self.fc1 = nn.Linear(256 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 3x32x32 -> 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 32x16x16 -> 64x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 64x8x8 -> 128x4x4
        x = F.relu(self.conv4(x))  # 128x4x4 -> 256x4x4 (בלי pool נוסף)

        x = x.flatten(1)  # 256*4*4 נוירונים
        x = F.relu(self.fc1(x))
        return self.fc2(x)
