import torch.nn as nn
import torch

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, s, p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

class InceptionA(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        c = c_out // 4
        self.b1 = ConvBNReLU(c_in, c, k=1, p=0)
        self.b2 = nn.Sequential(ConvBNReLU(c_in, c, k=1, p=0), ConvBNReLU(c, c, k=3, p=1))
        self.b3 = nn.Sequential(ConvBNReLU(c_in, c, k=1, p=0), ConvBNReLU(c, c, k=5, p=2))
        self.b4 = nn.Sequential(nn.MaxPool2d(3, stride=1, padding=1), ConvBNReLU(c_in, c, k=1, p=0))
    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)

class Meso4(nn.Module):
    def __init__(self, num_classes=1, img_size=256):
        super().__init__()
        self.features = nn.Sequential(
            ConvBNReLU(3, 8, k=3, p=1),
            nn.MaxPool2d(2,2),
            ConvBNReLU(8, 8, k=5, p=2),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.2),
            ConvBNReLU(8, 16, k=5, p=2),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.2),
            ConvBNReLU(16, 16, k=5, p=2),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*4*4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(-1)

class MesoInception4(nn.Module):
    def __init__(self, num_classes=1, img_size=256):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNReLU(3, 16, k=3, p=1),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(0.1)
        )
        self.inc1 = InceptionA(16, 32); self.pool1 = nn.MaxPool2d(2,2); self.drop1 = nn.Dropout2d(0.2)
        self.inc2 = InceptionA(32, 48); self.pool2 = nn.MaxPool2d(2,2); self.drop2 = nn.Dropout2d(0.3)
        self.inc3 = InceptionA(48, 64); self.pool3 = nn.MaxPool2d(2,2); self.drop3 = nn.Dropout2d(0.3)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten(),
            nn.Linear(64*4*4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.stem(x)
        x = self.inc1(x); x = self.pool1(x); x = self.drop1(x)
        x = self.inc2(x); x = self.pool2(x); x = self.drop2(x)
        x = self.inc3(x); x = self.pool3(x); x = self.drop3(x)
        x = self.head(x)
        return x.squeeze(-1)
