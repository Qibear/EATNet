# -*- coding: utf-8 -*-
# @Time    : 2023/10/12 21:12
# @Author  : debt


import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )



class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)


            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)

            elif pool_type == 'lse':
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial

        # 对于空间池化，则看情况使用
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


# Residual Attention Block (RAB)
class RAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CBAM(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
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


# ------------------------测试-------------------------------------------------
def Test1():
    a = torch.randn([2, 3, 384, 384])

    down1 = Down(3, 64)
    down2 = Down(64, 128)

    res1 = down1(a)
    print(res1.shape)  # torch.Size([2, 64, 192, 192])

    res2 = down2(res1)
    print(res2.shape)  # torch.Size([2, 128, 96, 96])


class EdgeAwareModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = Down(3, 64)
        self.down2 = Down(64, 128)

        # conv
        self.conv1 = conv3x3_bn_relu(128, 128)
        self.conv2 = conv3x3_bn_relu(128, 256, stride=2)

        # 将两个特征的channel转换成中间尺度，然后进行融合
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 32, 1)
        self.conv4 = nn.Conv2d(256, 32, 1)
        self.conv5 = conv3x3_bn_relu(32, 32)

        # RAB
        self.rab = RAB(32 * 2)

        # 输出前卷积操作
        self.conv6 = nn.Conv2d(32 * 2, 1, kernel_size=3, padding=1)

    def forward(self, input_depth, f1, f2):
        _, _, h, w = f1.size()

        input_depth = self.down1(input_depth)
        input_depth = self.down2(input_depth)  # ([2, 128, 96, 96])
        x1 = self.conv1(input_depth) + f1  # ([2, 128, 96, 96])
        x2 = self.conv2(x1) + f2

        # TODO 后续处理，完成后将与目标边缘图进行损失计算
        x1_m = self.relu(self.conv3(x1))  # ([2, 32, 96, 96])
        x2_m = self.relu(self.conv4(x2))  # ([2, 32, 48, 48])

        x2_m = F.interpolate(x2_m, size=(h, w), mode='bilinear', align_corners=True)
        edge = torch.cat([x1_m, x2_m], dim=1)
        edge = self.rab(edge)  # 2 64 96 96
        edge = self.conv6(edge)  # 2 1 96 96
        return edge


if __name__ == '__main__':
    a = torch.randn([2, 3, 384, 384])

    f1 = torch.randn([2, 128, 96, 96])
    f2 = torch.randn([2, 256, 48, 48])

    EG = EdgeAwareModule()
    a = EG(a, f1, f2)

    print(a.shape)
