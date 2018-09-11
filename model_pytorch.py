import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1,
                 activation=nn.ReLU(inplace=True)):
        super(ConvActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class ConvBNActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1,
                 activation=nn.ReLU(inplace=True)):
        super(ConvBNActivation, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1,
                 batch_norm=False, activation=nn.ReLU(inplace=True)):
        super(ConvBlock, self).__init__()
        conv = ConvBNActivation if batch_norm else ConvActivation
        self.block = nn.Sequential(
            conv(in_channels, out_channels, kernel_size,
                 stride, padding, dilation, activation),
            conv(out_channels, out_channels, kernel_size,
                 stride, padding, dilation, activation)
        )

    def forward(self, x):
        out = self.block(x)
        return out


class UpBlockWithSkip(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=1, dilation=1, up_mode='deconv',
                 batch_norm=False, activation=nn.ReLU(inplace=True)):
        assert up_mode in ('deconv', 'biupconv', 'nnupconv')
        super(UpBlockWithSkip, self).__init__()

        if up_mode == 'deconv':
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, padding=1)
        elif up_mode == 'biupconv':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2,
                            align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        elif up_mode == 'nnupconv':
            self.up = nn.Sequential(
                nn.Upsample(mode='nearest', scale_factor=2,
                            align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )

        self.conv_block = ConvBlock(
            out_channels * 2, out_channels, kernel_size,
            stride, padding, dilation, batch_norm, activation)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)

        return out


class DilatedUNet(nn.Module):

    def __init__(self, in_channels=3, classes=1, depth=3,
                 first_channels=44, padding=1,
                 bottleneck_depth=6, bottleneck_type='cascade',
                 batch_norm=False, up_mode='deconv',
                 activation=nn.ReLU(inplace=True)):

        assert bottleneck_type in ('cascade', 'parallel')
        super(DilatedUNet, self).__init__()

        self.depth = depth
        self.bottleneck_type = bottleneck_type

        conv = ConvBNActivation if batch_norm else ConvActivation

        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                ConvBlock(prev_channels, first_channels * 2**i, 3,
                          padding=padding, batch_norm=batch_norm,
                          activation=activation))
            prev_channels = first_channels * 2**i

        self.bottleneck_path = nn.ModuleList()
        for i in range(bottleneck_depth):
            bneck_in = prev_channels if i == 0 else prev_channels * 2
            self.bottleneck_path.append(
                conv(bneck_in, prev_channels * 2, 3,
                     dilation=2**i, padding=2**i, activation=activation))

        prev_channels *= 2

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UpBlockWithSkip(prev_channels, first_channels * 2**i, 3,
                                up_mode=up_mode, padding=padding,
                                batch_norm=batch_norm,
                                activation=activation))
            prev_channels = first_channels * 2**i

        self.last = nn.Conv2d(prev_channels, classes, kernel_size=1)

    def forward(self, x):
        bridges = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            bridges.append(x)
            x = F.avg_pool2d(x, 2)

        dilated_layers = []
        for i, bneck in enumerate(self.bottleneck_path):
            if self.bottleneck_type == 'cascade':
                x = bneck(x)
                dilated_layers.append(x.unsqueeze(-1))
            elif self.bottleneck_type == 'parallel':
                dilated_layers.append(bneck(x.unsqueeze(-1)))
        x = torch.cat(dilated_layers, dim=-1)
        x = torch.sum(x, dim=-1)

        for i, up in enumerate(self.up_path):
            x = up(x, bridges[-i-1])

        return self.last(x)
