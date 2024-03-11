from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# from fluid.core.nn.init import ICNR


class UNetDownBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, dis_mark=False):
        super(UNetDownBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.padding = padding
        if dis_mark:
            self.stride = 2
        else:
            self.stride = 1
        self.flag = dis_mark
        mid_size = (in_size + out_size) // 2

        block = []

        if batch_norm:
            block.append(nn.BatchNorm2d(in_size))
        block.append(nn.LeakyReLU(0.1, inplace=True))
        block.append(
            nn.Conv2d(
                in_size,
                mid_size,
                kernel_size=3,
                padding=int(padding),
                stride=self.stride,
                bias=not (batch_norm),
            )
        )
        if batch_norm:
            block.append(nn.BatchNorm2d(mid_size))
        block.append(nn.LeakyReLU(0.1, inplace=True))
        block.append(
            nn.Conv2d(
                mid_size,
                out_size,
                kernel_size=3,
                padding=int(padding),
                stride=self.stride,
                bias=not (batch_norm),
            )
        )
        self.block = nn.Sequential(*block)

    def center_crop(self, layer, target_size: List[int]):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x):
        out = self.block(x)
        if self.flag:
            return out
        temp = F.pad(x, (0, 0, 0, 0, 0, self.out_size - self.in_size))
        if self.padding is False:
            temp = self.center_crop(temp, out.shape[2:])
        return out + temp


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm):
        super(UNetUpBlock, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.padding = padding
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(
                in_size // 2, in_size // 2, kernel_size=2, stride=2
            )
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2, align_corners=False)
            )
        # elif up_mode == 'pixelshuffle':
        #     conv_shuffle = nn.Conv2d(in_size//2, in_size//2 * (2 ** 2), 3,
        #                              padding=1, bias=0)
        #     kernel = ICNR(conv_shuffle.weight, upscale_factor=2)
        #     conv_shuffle.weight.data.copy_(kernel)
        #     self.up = nn.Sequential(
        #                 conv_shuffle,
        #                 nn.PixelShuffle(2)
        #               )
        self.bridge_bn = nn.BatchNorm2d(in_size // 2)
        self.conv_block = UNetDownBlock(in_size, out_size, padding, batch_norm)

    def center_crop(self, layer, target_size: List[int]):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        if self.padding is False:
            bridge = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, self.bridge_bn(bridge)], 1)
        out = self.conv_block(out)
        return out


class fully_connected_layer(nn.Module):
    def __init__(self, in_channels):
        super(fully_connected_layer, self).__init__()
        self.fc = nn.ModuleList()
        self.fc.append(nn.Linear(in_channels, in_channels // 2))
        self.fc.append(nn.LeakyReLU(0.1, inplace=True))
        # drop out
        self.fc.append(nn.Dropout(p=0.5))
        self.fc.append(nn.Linear(in_channels // 2, 1))
        # self.fc.append(nn.Sigmoid())

    def forward(self, x):
        # x = torch.mean(x,3)
        # x = torch.mean(x,2)
        x = x.view(x.size(0), -1)
        for _, fc in enumerate(self.fc):
            x = fc(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        n_levels=5,
        n_channels=64,
        padding=True,
        batch_norm=False,
        up_mode="upsample",
    ):
        super(Generator, self).__init__()
        assert up_mode in ("upconv", "upsample", "pixelshuffle")

        out_ch = n_channels // 2**n_levels
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, out_ch // 2, kernel_size=3, padding=int(padding)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                out_ch // 2,
                out_ch,
                kernel_size=3,
                padding=int(padding),
                bias=not (batch_norm),
            ),
        )
        self.down_path = nn.ModuleList()
        for i in range(n_levels):
            in_ch = n_channels // 2 ** (n_levels - i)
            out_ch = n_channels // 2 ** (n_levels - i - 1)
            self.down_path.append(UNetDownBlock(in_ch, out_ch, padding, batch_norm))

        self.center = UNetDownBlock(n_channels, n_channels, padding, batch_norm)
        self.up_path = nn.ModuleList()
        for i in reversed(range(n_levels)):
            in_ch = n_channels // 2 ** (n_levels - i) * 4
            out_ch = n_channels // 2 ** (n_levels - i)
            self.up_path.append(
                UNetUpBlock(in_ch, out_ch, up_mode, padding, batch_norm)
            )

        self.last = nn.ModuleList()
        self.last.append(
            nn.Conv2d(
                out_ch, out_ch // 2, kernel_size=3, padding=1, bias=not (batch_norm)
            )
        )
        if batch_norm:
            self.last.append(nn.BatchNorm2d(out_ch // 2))
        self.last.append(nn.LeakyReLU(0.1, inplace=True))
        self.last.append(nn.Conv2d(out_ch // 2, out_channels, kernel_size=3, padding=1))
        self.last.append(nn.Tanh())

    def forward(self, x):
        blocks = []
        x = self.first(x)
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            x = F.avg_pool2d(x, 2)

        x = self.center(x)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        for _, fn in enumerate(self.last):
            x = fn(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels=1,
        n_levels=5,
        n_channels=64,
        image_size=256,
        padding=True,
        batch_norm=False,
    ):
        super(Discriminator, self).__init__()

        out_ch = n_channels // 2**n_levels
        self.first = nn.Sequential(
            nn.Conv2d(in_channels, out_ch // 2, kernel_size=3, padding=int(padding)),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                out_ch // 2,
                out_ch,
                kernel_size=3,
                padding=int(padding),
                bias=not (batch_norm),
            ),
        )
        self.down_path = nn.ModuleList()
        for i in range(n_levels):
            in_ch = n_channels // 2 ** (n_levels - i)
            out_ch = n_channels // 2 ** (n_levels - i - 1)
            self.down_path.append(UNetDownBlock(in_ch, out_ch, padding, batch_norm))

        self.center = UNetDownBlock(n_channels, n_channels, padding, batch_norm, True)
        self.last = fully_connected_layer(
            n_channels * (image_size // 2 ** (n_levels + 2)) ** 2
        )
        # Input dimension is channels*((image_size/2^(depth+2))^2)

    def forward(self, x):
        blocks = []
        x = self.first(x)
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            x = F.avg_pool2d(x, 2)

        x = self.center(x)
        return self.last(x)
