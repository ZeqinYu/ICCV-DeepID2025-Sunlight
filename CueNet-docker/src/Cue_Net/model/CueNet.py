import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from Cue_Net.model.encoder import get_backbone
from Cue_Net.model.EAM import EdgeAwareModule

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y

class PSPModule(nn.Module):
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                      for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                                         * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]
        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.Softmax(dim=1)):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, x):
        if self.activation is not None:
            return self.activation(self.conv(x))
        return self.conv(x)

class CueNet(nn.Module):
    def __init__(self, num_classes=1, backbone='backbone', pretrained=True, activation=nn.Sigmoid(), freeze_backbone=False):
        super(CueNet, self).__init__()
        self.freeze_backbone = freeze_backbone 
        if 'backbone' in backbone:
            self.backbone = get_backbone(model_name=backbone, pretrained=pretrained, num_classes=num_classes)
            feature_channels = self.backbone.dims
        else:
            pass
        fpn_out = feature_channels[0]
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        self.head = nn.Conv2d(fpn_out, num_classes, kernel_size=3, padding=1)
        self.cls_head = nn.Linear(feature_channels[-1], 1)
        self.cls_norm = nn.LayerNorm(feature_channels[-1], eps=1e-6)
        self.outc = OutConv(num_classes, num_classes, activation)
        self.eem = EdgeAwareModule()
        self._freeze_backbone() 

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        features = self.backbone(x)
        edge = self.eem(features[1], features[-1])
        edge = torch.sigmoid(edge)
        edge = F.interpolate(edge, size=input_size, mode='bilinear')
        score = self.cls_head(self.cls_norm(features[-1].mean([-2, -1])))
        score = torch.sigmoid(score)
        features[-1] = self.PPN(features[-1])
        d1 = self.FPN(features)
        x = self.head(d1)
        x = F.interpolate(x, size=input_size, mode='bilinear')
        out = self.outc(x)
        return out, score, edge, d1

    def _freeze_backbone(self):
        if self.freeze_backbone:
            self.backbone.eval()
            for params in self.backbone.parameters():
                params.requires_grad = False


