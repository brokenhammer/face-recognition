import torchvision
import torch.nn as nn
import torch.nn.functional as F
from config import cfg


class BaseBlock(nn.Module):
    def __init__(self, stride, c_in, expansion, c_out):
        super(BaseBlock, self).__init__()
        c_mid = c_in * expansion
        self.expand = nn.Conv2d(c_in, c_mid, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.trans = nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=stride, padding=1, groups=c_mid)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.squeeze = nn.Conv2d(c_mid, c_out, 1)
        self.bn3 = nn.BatchNorm2d(c_out)

        if c_in == c_out and stride == 1:
            self.shortcut = lambda x: x
        else:
            self.shortcut = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.expand(x)
        x = F.relu6(self.bn1(x), inplace=True)
        x = self.trans(x)
        x = F.relu6(self.bn2(x), inplace=True)
        x = self.squeeze(x)
        x = self.bn3(x)
        x = x + residual

        return x


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,padding=1)
        self.stage1 = self.bottleneck(32, 16, 1, 1, 1)
        self.stage2 = self.bottleneck(16, 24, 6, 2, 2)
        self.stage3 = self.bottleneck(24, 32, 6, 2, 3)
        self.stage4 = self.bottleneck(32, 64, 6, 2, 4)
        self.stage5 = self.bottleneck(64, 96, 6, 1, 3)
        self.stage6 = self.bottleneck(96, 160, 6, 2, 3)
        self.stage7 = self.bottleneck(160, 320, 6, 1, 1)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0)
        self.last_bn1 = nn.BatchNorm2d(1280)
        self.last_drop = nn.Dropout(0.5)
        self.last_fc = nn.Linear(7*7*1280, cfg.face_feat_size)
        self.last_bn2 = nn.BatchNorm1d(cfg.face_feat_size)

    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv2(x)
        x = self.last_bn1(x)
        x = self.last_drop(x)
        x = x.view(bs, -1)
        x = self.last_fc(x)
        x = self.last_bn2(x)

        return x

    def bottleneck(self, c_in, c_out, expansion, stride, repeat):
        layers = []
        layers.append(BaseBlock(stride, c_in, expansion, c_out))
        for i in range(repeat-1):
            layers.append(BaseBlock(1, c_out, expansion, c_out))

        return nn.Sequential(*layers)
