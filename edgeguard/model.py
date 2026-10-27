"""Base perception models. Every model returns (logits, penultimate_features)
from ONE forward pass, so all uncertainty signals come for free."""
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def to_tensor(x_uint8):
    """(N,32,32,3) uint8 numpy -> normalised float tensor (N,3,32,32)."""
    t = torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float() / 255.0
    return (t - MEAN) / STD


def _block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class EdgeNet(nn.Module):
    """Compact VGG-style CNN (~0.29M params) for CPU / micro-edge runs."""
    def __init__(self, num_classes=10, width=32, p_drop=0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            _block(3, w), _block(w, w), nn.MaxPool2d(2),
            _block(w, 2 * w), _block(2 * w, 2 * w), nn.MaxPool2d(2),
            _block(2 * w, 4 * w), _block(4 * w, 4 * w), nn.MaxPool2d(2),
        )
        self.feat_dim = 4 * w
        self.drop = nn.Dropout(p_drop)  # also used for the MC-Dropout baseline
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x):
        h = self.features(x)
        f = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(self.drop(f)), f


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + self.sc(x))


class ResNet18(nn.Module):
    """CIFAR ResNet-18 (11.2M params) — use this on a GPU (Colab) for final numbers."""
    def __init__(self, num_classes=10, p_drop=0.3):
        super().__init__()
        self.c1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.b1 = nn.BatchNorm2d(64)
        cfg = [(64, 1), (128, 2), (256, 2), (512, 2)]
        layers, cin = [], 64
        for cout, s in cfg:
            layers += [BasicBlock(cin, cout, s), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.feat_dim = 512
        self.drop = nn.Dropout(p_drop)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        h = self.layers(F.relu(self.b1(self.c1(x))))
        f = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(self.drop(f)), f


def build_model(arch="edgenet", num_classes=10):
    if arch == "edgenet":
        return EdgeNet(num_classes)
    if arch == "resnet18":
        return ResNet18(num_classes)
    raise ValueError(arch)


@torch.no_grad()
def infer(model, x_uint8, bs=500, device="cpu"):
    """Returns logits (N,C) and features (N,D) as numpy."""
    model.eval()
    L, Fs = [], []
    for i in range(0, len(x_uint8), bs):
        lg, f = model(to_tensor(x_uint8[i:i + bs]).to(device))
        L.append(lg.cpu()), Fs.append(f.cpu())
    return torch.cat(L).numpy(), torch.cat(Fs).numpy()
