# -*- coding: utf-8 -*-
# @Time    : 2023/10/17 13:58
# @Author  : debt

import torch
import torch.nn as nn


# TODO 重构建融合块
class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.single_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            SingleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


# Cross modal feature enhancement fusion module
class CFEM_Right(nn.Module):

    def __init__(self, in_channel=1024):
        super(CFEM_Right, self).__init__()

        self.depth_channel_attention = ChannelAttention(in_channel)
        self.rgb_channel_attention = ChannelAttention(in_channel)
        self.rd_spatial_attention = SpatialAttention()
        self.rgb_spatial_attention = SpatialAttention()
        self.depth_spatial_attention = SpatialAttention()

        self.down = Down(in_channel // 2, in_channel)

    def forward(self, r, d, r_before, d_before):
        # 输入为 [b 1024 12 12] 和 [b 512 24 24]
        assert r.shape == d.shape, "rgb and depth should have same size"

        r_before = self.down(r_before)
        d_before = self.down(d_before)

        r = r + r_before
        d = d + d_before
        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        r_f = r * sa
        d_f = d * sa
        r_ca = self.rgb_channel_attention(r_f)
        d_ca = self.depth_channel_attention(d_f)
        r_out = r * r_ca
        d_out = d * d_ca

        # TODO 做交叉特征融合
        mul_fea = r_out * d_out
        add_fea = r_out + d_out
        fuse_fea = torch.cat([mul_fea, add_fea], dim=1)  # 输出 [b 2048 12 12]

        return fuse_fea


class CFEM_Mid(nn.Module):

    def __init__(self, in_channel=512):  # in_channel ——> [256 512]
        super(CFEM_Mid, self).__init__()

        self.depth_channel_attention = ChannelAttention(in_channel)
        self.rgb_channel_attention = ChannelAttention(in_channel)
        self.rd_spatial_attention = SpatialAttention()
        self.rgb_spatial_attention = SpatialAttention()
        self.depth_spatial_attention = SpatialAttention()
        self.down = Down(in_channel // 2, in_channel)

    def forward(self, r, d, r_before, d_before):
        assert r.shape == d.shape, "rgb and depth should have same size"

        r_before = self.down(r_before)
        d_before = self.down(d_before)
        r = r + r_before
        d = d + d_before
        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        r_f = r * sa
        d_f = d * sa
        r_ca = self.rgb_channel_attention(r_f)
        d_ca = self.depth_channel_attention(d_f)
        r_out = r * r_ca
        d_out = d * d_ca
        mul_fea = r_out * d_out
        add_fea = r_out + d_out
        fuse_fea = torch.cat([mul_fea, add_fea], dim=1)

        # TODO 中间模块存在三输入
        return r_out, d_out, fuse_fea


class CFEM_Left(nn.Module):
    def __init__(self, in_channel=128):
        super(CFEM_Left, self).__init__()
        self.depth_channel_attention = ChannelAttention(in_channel)
        self.rgb_channel_attention = ChannelAttention(in_channel)
        self.rd_spatial_attention = SpatialAttention()
        self.rgb_spatial_attention = SpatialAttention()
        self.depth_spatial_attention = SpatialAttention()

    def forward(self, r, d):
        assert r.shape == d.shape, "rgb and depth should have same size"
        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        r_f = r * sa
        d_f = d * sa
        r_ca = self.rgb_channel_attention(r_f)
        d_ca = self.depth_channel_attention(d_f)
        r_out = r * r_ca
        d_out = d * d_ca
        mul_fea = r_out * d_out
        add_fea = r_out + d_out
        fuse_fea = torch.cat([mul_fea, add_fea], dim=1)

        return r_out, d_out, fuse_fea


if __name__ == '__main__':
    a = torch.randn([2, 1024, 12, 12])
    b = torch.randn([2, 512, 24, 24])
    c = torch.randn([2, 256, 48, 48])  # 这个是第二个中间模块，暂时不做测试
    d = torch.randn([2, 128, 96, 96])

    # TODO 测试右边的特征融合模块
    r_net = CFEM_Right(1024)
    r_res = r_net(a, a, b, b)

    # TODO 测试中间融合模块
    m_net = CFEM_Mid(512)
    m_res1, m_res2, m_res3 = m_net(b, b, c, c)

    # TODO 测试左边融合模块
    l_net = CFEM_Left(128)
    l_res1, l_res2, l_res3 = l_net(d, d)

    print()
