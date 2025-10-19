import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBR, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class CBR_reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(CBR_reduction, self).__init__()
        self.reduce_block = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce_block(x)

class EdgeAwareModule(nn.Module):
    def __init__(self, ):
        super(EdgeAwareModule, self).__init__()
        self.low_CBR = CBR_reduction(384, 384//2) 
        self.high_CBR = CBR_reduction(1536, 1536//2)
        self.block = nn.Sequential(
            ConvBR(384//2 + 1536//2, 1536//2, 3, padding=1),
            ConvBR(1536//2, 1536//4, 3, padding=1),
            nn.Conv2d(1536//4, 1, 1))

    def forward(self, low, high):
        size = low.size()[2:]
        low = self.low_CBR(low)
        high = self.high_CBR(high)
        high = F.interpolate(high, size, mode='bilinear', align_corners=False)
        out = torch.cat((high, low), dim=1)
        out = self.block(out)
        return out



