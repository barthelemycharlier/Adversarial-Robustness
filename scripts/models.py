import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------
# CIFAR-10 ResNet-18 (paper version, with selectable activation)
# ------------------------------------------------------------

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3,
                     stride=stride, padding=1, bias=False)

class BasicBlock_CIFAR(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, activation="relu"):
        super().__init__()
        self.act_type = activation.lower()
        self.act = nn.GELU() if self.act_type == "gelu" else nn.ReLU(inplace=True)

        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = x if self.shortcut is None else self.shortcut(x)
        out += shortcut
        out = self.act(out)
        return out


class ResNet18CIFAR(nn.Module):
    """
    CIFAR-10 version of ResNet-18 (used in the MEANSPARSE paper).
    Activation selectable: relu or gelu.
    """

    def __init__(self, num_classes=10, activation="gelu"):
        super().__init__()

        self.act_type = activation.lower()
        self.act = nn.GELU() if self.act_type == "gelu" else nn.ReLU(inplace=True)

        self.in_planes = 64

        # CIFAR-style 3x3 conv stem (NO maxpool)
        self.conv1 = conv3x3(3, 64)
        self.bn1 = nn.BatchNorm2d(64)

        # Stages
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        layers = []
        layers.append(BasicBlock_CIFAR(self.in_planes, planes,
                                       stride=stride, activation=self.act_type))
        self.in_planes = planes
        for _ in range(1, num_blocks):
            layers.append(BasicBlock_CIFAR(planes, planes,
                                           stride=1, activation=self.act_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


class CNNBackbone(nn.Module):
    """
    CNN with three convolutional blocks and a fully connected head.
    Returns log-probabilities (log_softmax) for NLLLoss.
    """
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # Block 1
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1_1   = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn1_2   = nn.BatchNorm2d(32)
        # Block 2
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2_1   = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2_2   = nn.BatchNorm2d(64)
        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3_1   = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3_2   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.5)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool(x)  # 32 -> 16
        # Block 2
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool(x)  # 16 -> 8
        # Block 3
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool(x)  # 8 -> 4

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class WideBasic(nn.Module):
    """
    Basic block for WideResNet (CIFAR version).
    Inspired by wide-resnet.pytorch.
    """
    def __init__(self, in_planes: int, out_planes: int, stride: int,
                 dropRate: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(
            out_planes, out_planes,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.dropRate = dropRate

        if in_planes != out_planes or stride != 1:
            self.shortcut = nn.Conv2d(
                in_planes, out_planes,
                kernel_size=1, stride=stride, padding=0, bias=False
            )
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = F.relu(out, inplace=True)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out, inplace=True)

        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)

        out = self.conv2(out)

        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x

        return out + residual


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers: int, in_planes: int, out_planes: int,
                 block: nn.Module, stride: int, dropRate: float) -> None:
        super().__init__()
        layers = []
        for i in range(nb_layers):
            s = stride if i == 0 else 1
            inp = in_planes if i == 0 else out_planes
            layers.append(block(inp, out_planes, s, dropRate))
        self.layer = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class WideResNetBackbone(nn.Module):
    """
    WideResNet for CIFAR-10.
    depth = 6n+4 (e.g., 28), widen_factor e.g., 10 (WRN-28-10).
    Returns log_softmax for NLLLoss.
    """
    def __init__(
        self,
        depth: int = 28,
        widen_factor: int = 10,
        num_classes: int = 10,
        dropRate: float = 0.3
    ) -> None:
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        nChannels = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(
            3, nChannels[0],
            kernel_size=3, stride=1, padding=1, bias=False
        )

        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], WideBasic, stride=1,
            dropRate=dropRate
        )
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], WideBasic, stride=2,
            dropRate=dropRate
        )
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], WideBasic, stride=2,
            dropRate=dropRate
        )

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.fc = nn.Linear(nChannels[3], num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


# def build_backbone(arch: str = "cnn") -> nn.Module:
#     """
#     Build a backbone model based on the specified architecture.

#     arch:
#       - "cnn"        : CNNBackbone
#       - "wrn28x10"   : WideResNetBackbone(depth=28, widen_factor=10)
#     """
#     arch = arch.lower()
#     if arch == "cnn":
#         return CNNBackbone(num_classes=10)
#     elif arch == "wrn28x10":
#         return WideResNetBackbone(
#             depth=28, widen_factor=10, num_classes=10, dropRate=0.3
#         )
#     else:
#         raise ValueError(f"Unknown architecture '{arch}'. "
#                          f"Supported: 'cnn', 'wrn28x10'.")


def build_backbone(arch: str = "cnn", activation: str = "relu") -> nn.Module:
    arch = arch.lower()

    if arch == "cnn":
        return CNNBackbone(num_classes=10)

    elif arch == "wrn28x10":
        return WideResNetBackbone(
            depth=28, widen_factor=10, num_classes=10, dropRate=0.3
        )

    elif arch == "resnet18":
        return ResNet18CIFAR(num_classes=10, activation=activation)

    else:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         f"Supported: 'cnn', 'wrn28x10', 'resnet18'.")
