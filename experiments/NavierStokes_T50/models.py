import torch
import torch.nn as nn
from torchdiffeq import odeint


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()

        y = torch.cat(
            [
                self.pool_h(x),
                self.pool_w(x).permute(0, 1, 3, 2)
            ],
            dim=2
        )

        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)

        return x * self.conv_h(x_h).sigmoid() * self.conv_w(
            x_w.permute(0, 1, 3, 2)
        ).sigmoid()


class SDOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, modes1=2, modes2=3):
        super().__init__()

        self.modes1 = modes1
        self.modes2 = modes2

        self.weights1 = nn.Parameter(
            torch.complex(
                torch.randn(in_channels, out_channels, modes1, modes2) * 0.02,
                torch.randn(in_channels, out_channels, modes1, modes2) * 0.02
            )
        )

        self.weights2 = nn.Parameter(
            torch.complex(
                torch.randn(in_channels, out_channels, modes1, modes2) * 0.02,
                torch.randn(in_channels, out_channels, modes1, modes2) * 0.02
            )
        )

    def forward(self, x):
        x_ft = torch.fft.rfft2(x.float())
        out_ft = torch.zeros_like(x_ft)

        out_ft[:, :, :self.modes1, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :self.modes1, :self.modes2],
            self.weights1.to(x_ft.device)
        )

        out_ft[:, :, -self.modes1:, :self.modes2] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -self.modes1:, :self.modes2],
            self.weights2.to(x_ft.device)
        )

        return torch.fft.irfft2(
            out_ft,
            s=(x.size(-2), x.size(-1))
        ).to(x.dtype)


class DyMemCell(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.conv3d = nn.Conv3d(
            channels * 2,
            channels * 4,
            kernel_size=(2, 3, 3),
            padding=(0, 1, 1)
        )

    def forward(self, x, h, c):
        combined = torch.stack([x, h], dim=2)

        gates = self.conv3d(
            torch.cat([combined, combined], dim=1)
        ).squeeze(2)

        i, f, g, o = torch.split(gates, self.channels, dim=1)

        c_next = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_next = torch.sigmoid(o) * torch.tanh(c_next)

        return h_next, c_next


class NeuroSD(nn.Module):
    def __init__(
        self,
        in_channels=1,
        hid_channels=128,
        out_frames=10,
        modes1=2,
        modes2=3
    ):
        super().__init__()

        self.hid_channels = hid_channels
        self.out_frames = out_frames

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 3, padding=1),
            CoordAtt(hid_channels, hid_channels),
            nn.GroupNorm(16, hid_channels),
            nn.SiLU(),
            nn.Conv2d(hid_channels, hid_channels, 3, stride=2, padding=1)
        )

        self.sdo_block = SDOBlock(
            hid_channels,
            hid_channels,
            modes1=modes1,
            modes2=modes2
        )

        self.dymem_layer1 = DyMemCell(hid_channels)
        self.dymem_layer2 = DyMemCell(hid_channels)

        self.neural_ode_func = nn.Sequential(
            nn.Conv2d(hid_channels, hid_channels, 3, padding=1),
            nn.GroupNorm(8, hid_channels),
            nn.SiLU(),
            nn.Conv2d(hid_channels, hid_channels, 3, padding=1),
            nn.GroupNorm(8, hid_channels),
            nn.SiLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hid_channels, hid_channels, 4, stride=2, padding=1),
            nn.Conv2d(hid_channels, hid_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hid_channels, in_channels, 3, padding=1)
        )

    def forward(self, x):
        b, t, _, _, _ = x.shape

        sample_z = self.encoder(x[:, 0])
        _, _, h_out, w_out = sample_z.shape

        h1 = torch.zeros(b, self.hid_channels, h_out, w_out).to(x.device)
        c1 = torch.zeros(b, self.hid_channels, h_out, w_out).to(x.device)
        h2 = torch.zeros(b, self.hid_channels, h_out, w_out).to(x.device)
        c2 = torch.zeros(b, self.hid_channels, h_out, w_out).to(x.device)

        for i in range(t):
            f_t = self.encoder(x[:, i])
            z_t = self.sdo_block(f_t) + f_t

            h1, c1 = self.dymem_layer1(z_t, h1, c1)
            h2, c2 = self.dymem_layer2(h1 + z_t, h2, c2)

        z_series = odeint(
            lambda tau, z: self.neural_ode_func(z),
            h2,
            torch.linspace(0, 1, self.out_frames + 1).to(x.device),
            method="rk4"
        )[1:]

        return torch.stack(
            [self.decoder(z_series[i]) for i in range(self.out_frames)],
            dim=1
        )