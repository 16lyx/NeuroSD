"""MovingMNIST forecast interfaces. See mmnist_vendor/sources.json for changes.

All forward methods consume only observed frames [B,T,C,H,W]. No target frames,
pretrained weights, output clipping or teacher forcing are used.
"""
import json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

VENDOR = Path(__file__).with_name('mmnist_vendor')


class ForecastModel(nn.Module):
    nfe = 0

    def __init__(self, args):
        super().__init__()
        self.tin = args.pre_seq_length
        self.tout = args.aft_seq_length
        self.channels = args.in_shape[1]
        self.height, self.width = args.in_shape[-2:]

    def pack(self, x):
        if x.ndim != 5 or tuple(x.shape[1:]) != (self.tin, self.channels, self.height, self.width):
            raise ValueError(f'Unexpected input shape {tuple(x.shape)}')
        return x.reshape(x.shape[0], self.tin * self.channels, self.height, self.width)

    def unpack(self, y):
        return y.reshape(y.shape[0], self.tout, self.channels, self.height, self.width)


class EarthformerForecast(ForecastModel):
    def __init__(self, args):
        super().__init__(args)
        from .mmnist_vendor.earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
        cfg = json.loads((VENDOR / 'earthformer_mmnist_model.json').read_text())
        cfg['input_shape'] = [self.tin, self.height, self.width, self.channels]
        cfg['target_shape'] = [self.tout, self.height, self.width, self.channels]
        # Model architecture is the official moving_mnist/cfg.yaml, not trainer defaults.
        self.network = CuboidTransformerModel(**cfg)

    def forward(self, x):
        self.pack(x)  # input validation
        return self.network(x.permute(0, 1, 3, 4, 2).contiguous()).permute(0, 1, 4, 2, 3).contiguous()


class ResNetForecast(ForecastModel):
    def __init__(self, args):
        super().__init__(args)
        from .mmnist_vendor.resnet_reference import ResNet, BasicBlock
        net = ResNet(BasicBlock, [2, 2, 2, 2])
        net.conv1 = nn.Conv2d(self.tin * self.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.encoder = nn.Sequential(*list(net.children())[:-2])
        # Explicit prediction adaptation: remove classification pool/fc, project and resize.
        self.projection = nn.Conv2d(512, self.tout * self.channels, kernel_size=1)

    def forward(self, x):
        z = self.projection(self.encoder(self.pack(x)))
        return self.unpack(F.interpolate(z, size=(self.height, self.width), mode='bilinear', align_corners=False))


class UNetForecast(ForecastModel):
    def __init__(self, args):
        super().__init__(args)
        from .mmnist_vendor.unet_reference import UNetOrg
        # Official CNO authors' published U-Net baseline, including BN and same padding.
        # This is NOT claimed to be the original U-Net 2015 Caffe implementation.
        self.network = UNetOrg(self.tin * self.channels, self.tout * self.channels, channels=args.unet_channels, bilinear=False)

    def forward(self, x):
        return self.unpack(self.network(self.pack(x)))


class ViTBlock(nn.Module):
    """PyTorch port of the official Flax pre-LN encoder, not a checkpoint converter."""
    def __init__(self, dim, heads, mlp_dim, dropout, attention_dropout):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.attention_output = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        weights = self.attention_dropout(((q * (d // self.heads) ** -0.5) @ k.transpose(-2, -1)).softmax(dim=-1))
        y = (weights @ v).transpose(1, 2).reshape(b, n, d)
        x = x + self.drop(self.attention_output(y))
        y = self.fc2(self.drop(F.gelu(self.fc1(self.norm2(x)), approximate='tanh')))
        return x + self.drop(y)


class ViTForecast(ForecastModel):
    def __init__(self, args):
        super().__init__(args)
        p, d = args.vit_patch_size, args.vit_dim
        if self.height % p or self.width % p:
            raise ValueError('Image size must be divisible by patch size')
        self.p = p
        n = self.height // p * (self.width // p)
        self.embedding = nn.Conv2d(self.tin * self.channels, d, kernel_size=p, stride=p)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.position = nn.Parameter(torch.empty(1, n + 1, d))
        nn.init.normal_(self.position, std=0.02)
        self.drop = nn.Dropout(args.vit_dropout)
        self.blocks = nn.Sequential(*[ViTBlock(d, args.vit_heads, args.vit_mlp_dim, args.vit_dropout, args.vit_attention_dropout) for _ in range(args.vit_depth)])
        self.norm = nn.LayerNorm(d, eps=1e-6)
        # Explicit forecast head: each patch token predicts its entire future patch.
        self.head = nn.Linear(d, self.tout * self.channels * p * p)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.normal_(module.bias, std=1e-6)

    def forward(self, x):
        z = self.embedding(self.pack(x)).flatten(2).transpose(1, 2)
        z = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), z], dim=1)
        z = self.norm(self.blocks(self.drop(z + self.position)))[:, 1:]
        z = self.head(z)
        gh, gw, p = self.height // self.p, self.width // self.p, self.p
        return z.reshape(x.shape[0], gh, gw, self.tout, self.channels, p, p).permute(0, 3, 4, 1, 5, 2, 6).reshape(x.shape[0], self.tout, self.channels, self.height, self.width)


class CNOForecast(ForecastModel):
    def __init__(self, args):
        super().__init__(args)
        if self.height != self.width:
            raise ValueError('Classic CNO requires a square grid here')
        from .mmnist_vendor.cno_classic.CNOModule import CNO
        from .mmnist_vendor.cno_classic.training import filtered_networks
        if args.cno_filter_impl not in ('ref', 'cuda'):
            raise ValueError('cno_filter_impl must be ref or cuda')
        filtered_networks.FILTER_IMPL = args.cno_filter_impl
        self.network = CNO(in_dim=self.tin * self.channels, in_size=self.height,
                           out_dim=self.tout * self.channels, out_size=self.height,
                           N_layers=args.cno_layers, N_res=args.cno_res,
                           N_res_neck=args.cno_neck, channel_multiplier=args.cno_channels,
                           activation='cno_lrelu', batch_norm=True)

    def forward(self, x):
        return self.unpack(self.network(self.pack(x)))


class FNOForecast(ForecastModel):
    def __init__(self, args):
        super().__init__(args)
        from .mmnist_vendor.fno_reference import Net2d
        if self.tin != 10 or self.channels != 1:
            raise ValueError('Pinned original FNO reference expects ten single-channel history frames')
        self.network = Net2d(args.fno_modes, args.fno_width)
        gx = torch.linspace(0, 1, self.height).reshape(1, self.height, 1, 1).expand(1, self.height, self.width, 1)
        gy = torch.linspace(0, 1, self.width).reshape(1, 1, self.width, 1).expand(1, self.height, self.width, 1)
        self.register_buffer('grid', torch.cat([gx, gy], dim=-1), persistent=False)

    def forward(self, x):
        self.pack(x)
        history = x[:, :, 0].permute(0, 2, 3, 1).contiguous()
        outputs = []
        for _ in range(self.tout):
            y = self.network(torch.cat([history, self.grid.expand(x.shape[0], -1, -1, -1)], dim=-1))
            outputs.append(y.permute(0, 3, 1, 2))
            # No teacher forcing, and do not detach: gradients traverse the full rollout.
            history = torch.cat([history[..., 1:], y], dim=-1)
        return torch.stack(outputs, dim=1)


MODELS = dict(earthformer=EarthformerForecast, resnet=ResNetForecast,
              unet=UNetForecast, vit=ViTForecast, cno=CNOForecast, fno=FNOForecast)


def build_forecast_model(args):
    return MODELS[args.method.lower()](args)
