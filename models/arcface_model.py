import torch
import torchvision
import torch.nn as nn
from config import cfg
from torch.nn import Parameter
import torch.nn.functional as F
from .mobilenet_model import MobileNet
from .resnet import resnet_face18


class Classifier(nn.Module):
    def __init__(self, class_num, feat=False):
        super(Classifier, self).__init__()
        self.backbone = MobileNet()
        #self.backbone = resnet_face18(use_se=False)
        self.normlinear = NormLinear(cfg.face_feat_size, class_num)
        self.feat = feat

    def forward(self, x):
        x = self.backbone(x)
        if self.feat:
            return x
        x = self.normlinear(x)
        return x


class ArcSoftmax(nn.Module):
    def __init__(self, gamma=0, reduction=True):
        super(ArcSoftmax, self).__init__()
        self.gamma = gamma # focal loss parameter
        self.reduction = reduction

    def forward(self, input, labels):
        # size of x ((n, class_num), (n, c_out)) = (cos, cosm)
        # size of labels (n)
        cos, cosm = input
        n = cos.shape[0]
        batch_inds = torch.LongTensor(range(n))
        one_hot = torch.zeros_like(cos, device='cuda')
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        #cos[batch_inds, labels] = cosm[batch_inds, labels]
        cos_phi = one_hot * cosm + ((1.0 - one_hot) * cos)
        p = F.softmax(cos_phi, 1)
        pt = p[batch_inds, labels]
        focal_loss = - (1 - pt.data) ** self.gamma * torch.log(pt)
        if self.reduction:
            focal_loss = focal_loss.mean()
        return focal_loss


class NormLinear(nn.Module):
    def __init__(self, c_in, c_out, s=30, m=0.5):
        super(NormLinear, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.s = s
        self.m = m
        self.w = Parameter(torch.Tensor(c_out, c_in))
        torch.nn.init.xavier_uniform_(self.w)

    def forward(self, x):
        #TODO: add renorm to self.w
        # size of x (n, c_in)
        # size of w (c_out, c_in)
        # size of output (n, c_out)
        cos = F.linear(F.normalize(x), F.normalize(self.w))
        theta = torch.acos(cos)
        # easy margin ???
        cosm = torch.where(cos > 0, torch.cos(theta + self.m), cos)

        return (self.s * cos, self.s * cosm)