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


class DimensionalReduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DimensionalReduction, self).__init__()
        self.reduce = nn.Sequential(
            ConvBR(in_channel, out_channel, 3, padding=1),
            ConvBR(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class EdgeEstimationModule(nn.Module):
    def __init__(self, ):
        super(EdgeEstimationModule, self).__init__()
        self.reduce3 = DimensionalReduction(384, 384//2)  # 56 160 448    [128, 320, 512]
        self.reduce5 = DimensionalReduction(1536, 1536//2)
        self.block = nn.Sequential(
            ConvBR(384//2 + 1536//2, 1536//2, 3, padding=1),
            ConvBR(1536//2, 1536//4, 3, padding=1),
            nn.Conv2d(1536//4, 1, 1))

    def forward(self, x3, x5):
        size = x3.size()[2:]
        print(size)  # torch.Size([64, 64])
        x3 = self.reduce3(x3)
        print(x3.shape)  # torch.Size([4, 192, 64, 64])
        x5 = self.reduce5(x5)
        print(x5.shape)  # torch.Size([4, 768, 16, 16])
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=False)
        print(x5.shape)  # torch.Size([4, 768, 64, 64])
        out = torch.cat((x5, x3), dim=1)
        print(out.shape)  # torch.Size([4, 960, 64, 64])
        out = self.block(out)

        return out

if __name__ == '__main__':
    """
    torch.Size([4, 192, 128, 128])
    torch.Size([4, 384, 64, 64])
    torch.Size([4, 768, 32, 32])
    torch.Size([4, 1536, 16, 16])
    """
    input1 = torch.ones((4, 384, 64, 64))
    input2 = torch.ones((4, 1536, 16, 16))
    eer = EdgeEstimationModule()
    output = eer(input1, input2)
    print(output.shape)
