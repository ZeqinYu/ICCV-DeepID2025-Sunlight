import os
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

try:
    from .utils import LayerNorm, GRN
except:
    from utils import LayerNorm, GRN


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self, in_chans=3, num_classes=1000, use_cls_to_train=False,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., head_init_scale=1., to_cls=False
                 ):
        super().__init__()
        self.depths = depths
        self.dims = dims
        self.use_cls_to_train = use_cls_to_train
        self.to_cls = to_cls
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))  # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        if self.to_cls:
            x = self.forward_features(x)
            x = self.head(x)
            return x
        _, C, H, W = x.shape
        features = []

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        # if self.use_cls_to_train:
        #     score = self.head(self.norm(x.mean([-2, -1])))
        #     return features, score
        return features

    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.head(x)
    #     return x


def convnextv2_atto(pretrained=False, num_classes=1000, **kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    if pretrained:
        url = model_urls['convnextv2_atto']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
        model.head = nn.Linear(model.dims[-1], num_classes)

    return model


def convnextv2_large(pretrained=False, num_classes=1000, in_22k=False, img_size=384, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        model_name = 'convnextv2_large'
     
        model_name += '_22k' if in_22k else '_1k'
        model_name += '_384' if img_size >= 384 else '_224'
        url = model_urls[model_name]
        # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        checkpoint = torch.load(url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        model.head = nn.Linear(model.dims[-1], num_classes)
    return model

model_urls_oneline = {
    'convnextv2_large_22k_224': 'https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_224_ema.pt',
}

def convnextv2_base(pretrained=False, num_classes=1000, in_22k=False, img_size=384, **kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        model_name = 'convnextv2_base'
        model_name += '_22k' if in_22k else '_1k'
        model_name += '_384' if img_size >= 384 else '_224'
        url = model_urls_base[model_name]
        checkpoint = torch.load(url, map_location='cpu')
        # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
        # model.head = nn.Linear(model.dims[-1], num_classes)
    return model


model_urls = {
    'convnextv2_large_22k_224': '/app/src/Cue_Net/ck/convnextv2_large_22k_224_ema.pt'
}
# model_urls = {
#     'convnextv2_large_22k_224': '/home/tianye/code/baseline-docker-maincnet/src/Cue_Net/convnextv2_large_22k_224_ema.pt'
# }
model_urls_base = {
    'convnextv2_base_22k_224': '/app/src/Cue_Net/ck/convnextv2_base_22k_224_ema.pt'
}
# model_urls = {
#     'convnextv2_large_22k_224': './Cue_Net/convnextv2_large_22k_224_ema.pt'
# }

def get_convnextv2(model_name='convnextv2_atto', pretrained=False, **kwargs):
    if len(model_name.split('_'))>2:
        lines = model_name.rsplit('_', 2)
        model_name = lines[0]
        in_22k = True if lines[1]=='22k' else False
        img_size = eval(lines[2])

    if model_name == 'convnextv2_atto':
        model = convnextv2_atto(pretrained=pretrained, **kwargs)
    elif model_name == 'convnextv2_large':
        model = convnextv2_large(pretrained=pretrained, in_22k=in_22k, img_size=img_size, **kwargs)
    elif model_name == 'convnextv2_base':
        model = convnextv2_base(pretrained=pretrained, in_22k=in_22k, img_size=img_size, **kwargs)

    else:
        raise NotImplementedError(f'Unknown model: {model_name}')
    return model

