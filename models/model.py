from models.SwinT import SwinTransformer
from models.AFEM import AFEM
from models.EAM import EdgeAwareModule
from models.CFEM import CFEM_Left, CFEM_Mid, CFEM_Right
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from models.ConvMoudle import GhostConv

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def gsconv3x3(in_planes, out_planes):
    return GhostConv(in_planes, out_planes, k=3)


def gsconv3x3_bn_relu(in_planes, out_planes):
    return nn.Sequential(
        gsconv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


# Multi scale feature aggregation module
class MFAM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MFAM, self).__init__()

        # 双线性插值上采样
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = gsconv3x3_bn_relu(in_ch, out_ch)
        self.aff = AFF(out_ch)

    def forward(self, fuse_high, fuse_low):
        fuse_high = self.up2(fuse_high)
        fuse_high = self.conv(fuse_high)
        fe_decode = self.aff(fuse_high, fuse_low)

        return fe_decode


class eca_layer(nn.Module):
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        return y.expand_as(x)


class AFF(nn.Module):
    def __init__(self, channels=64, r=4):
        super(AFF, self).__init__()
        self.conv_sa = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, dilation=1,
                                 groups=channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_ca = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        sa_x = self.conv_sa(xa)
        y = self.avg_pool(xa)
        ca_x = self.conv_ca(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        xlg = sa_x + ca_x
        wei = self.sigmoid(xlg)

        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo


# 自校准卷积
class SCConv(nn.Module):
    def __init__(self, planes, pooling_r=4):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            gsconv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.k3 = nn.Sequential(
            gsconv3x3(planes, planes),
            nn.BatchNorm2d(planes),
        )
        self.k4 = nn.Sequential(
            gsconv3x3(planes, planes),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SEModel(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.size()
        y = self.avg_pool(x).view(bs, c)
        y = self.fc(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


# Attention Connection
class AC(nn.Module):
    def __init__(self, channel=256):
        super().__init__()

        self.conv = gsconv3x3_bn_relu(channel, channel * 2)
        self.conv1x1 = nn.Conv2d(channel, channel * 2, 1)

        self.se = SEModel(channel)
        self.sconv = SCConv(channel * 2)

    def forward(self, x_left, x_up):
        f_use = x_up + x_left
        f_use = self.se(f_use)
        f_use_m = self.conv(f_use)
        f_use_r = self.conv1x1(f_use)

        f_use = f_use_r + f_use_m
        f_use = self.sconv(f_use)  # 自校准卷积

        return f_use


class CascadedDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder_left = DecoderLeft()
        self.decoder_right = DecoderRight()

        self.ac = AC(256)
        self.ac2 = AC(512)
        self.ac3 = AC(1024)

        self.down = nn.MaxPool2d(2)

    def forward(self, fuse1, fuse2, fuse3, fuse4):
        left_mfam_1_out, left_mfam_2_out, left_mfam_3_out = self.decoder_left(fuse1, fuse2, fuse3, fuse4)
        ac_out1 = self.ac(left_mfam_1_out, left_mfam_1_out)
        ac_out1 = self.down(ac_out1)  # [2 512 48 48]
        ac_out2 = self.ac2(left_mfam_2_out, ac_out1)
        ac_out2 = self.down(ac_out2)  # [2 1024 24 24]
        ac_out3 = self.ac3(left_mfam_3_out, ac_out2)
        ac_out3 = self.down(ac_out3)

        mfam_1_out = self.decoder_right(ac_out1, ac_out2, ac_out3, fuse1, fuse2, fuse3, fuse4)
        return left_mfam_1_out, mfam_1_out


class DecoderLeft(nn.Module):
    def __init__(self):
        super(DecoderLeft, self).__init__()

        # 三对mfam结构
        self.mfam_1 = MFAM(512, 256)
        self.mfam_2 = MFAM(1024, 512)
        self.mfam_3 = MFAM(2048, 1024)

    def forward(self, f_u_1, f_u_2, f_u_3, f_u_4):
        mfam_3_out = self.mfam_3(f_u_4, f_u_3)
        mfam_2_out = self.mfam_2(mfam_3_out, f_u_2)
        mfam_1_out = self.mfam_1(mfam_2_out, f_u_1)

        return mfam_1_out, mfam_2_out, mfam_3_out,


class DecoderRight(nn.Module):
    def __init__(self):
        super(DecoderRight, self).__init__()

        self.mfam_1 = MFAM(512, 256)
        self.mfam_2 = MFAM(1024, 512)
        self.mfam_3 = MFAM(2048, 1024)

        self.conv_1 = nn.Conv2d(512, 256, 1)
        self.conv_2 = nn.Conv2d(1024, 512, 1)
        self.conv_3 = nn.Conv2d(2048, 1024, 1)

    def forward(self, ac1, ac2, ac3, f_u_1, f_u_2, f_u_3, f_u_4):
        ac1_inter = F.interpolate(ac1, size=(96, 96), mode='bilinear')
        ac1_inter = self.conv_1(ac1_inter)
        ac1_inter = ac1_inter + f_u_1  # TODO

        ac2_inter = F.interpolate(ac2, size=(48, 48), mode='bilinear')
        ac2_inter = self.conv_2(ac2_inter)
        ac2_inter = ac2_inter + f_u_2

        ac3_inter = F.interpolate(ac3, size=(24, 24), mode='bilinear')
        ac3_inter = self.conv_3(ac3_inter)
        ac3_inter = ac3_inter + f_u_3

        # TODO 从小往上看
        mfam_3_out = self.mfam_3(f_u_4, ac3_inter)
        mfam_2_out = self.mfam_2(mfam_3_out, ac2_inter)
        mfam_1_out = self.mfam_1(mfam_2_out, ac1_inter)

        return mfam_1_out


class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()

        # TODO backbone不做修改
        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.afem = AFEM(1024, 1024)

        # TODO 使用自己的多模态特征融合模块
        self.mfcm_1 = CFEM_Left(128)
        self.mfcm_2 = CFEM_Mid(256)
        self.mfcm_3 = CFEM_Mid(512)
        self.mfcm_4 = CFEM_Right(1024)

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.conv256_32 = conv3x3_bn_relu(256, 32)
        self.conv64_1 = conv3x3(64, 1)

        # TODO 生成边缘图
        self.eam = EdgeAwareModule()

        self.edge_feature = conv3x3_bn_relu(1, 32)
        self.fuse_edge_sal = conv3x3(32, 1)
        self.up_edge = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            conv3x3(32, 1)
        )

        self.relu = nn.ReLU(True)
        self.cascaded_decoder = CascadedDecoder()

    def forward(self, x, d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        rgb1_u = rgb_list[0]  # [2 128 96 96]
        rgb2_u = rgb_list[1]  # [2 256 48 48]
        rgb3_u = rgb_list[2]  # [2 512 24 24]
        rgb4_u = rgb_list[3]  # [2 1024 12 12]


        d1_u = depth_list[0]
        d2_u = depth_list[1]
        d3_u = depth_list[2]
        d4_u = depth_list[3]

        # [2 128 96 96] ——> [2 256 96 96]
        l_f1, l_f2, f_u_1 = self.mfcm_1(rgb1_u, d1_u)

        # [2 256 48 48] ——> [2 512 48 48]
        m1_f1, m1_f2, f_u_2 = self.mfcm_2(rgb2_u, d2_u, l_f1, l_f2)

        # [2 512 24 24] ——> [2 1024 24 24]
        m2_f1, m2_f2, f_u_3 = self.mfcm_3(rgb3_u, d3_u, m1_f1, m1_f2)

        # [2 1024 12 12] ——> [2 2048 12 12]
        f_u_4 = self.mfcm_4(rgb4_u, d4_u, m2_f1, m2_f2)

        end_fuse_s1, end_fuse_s = self.cascaded_decoder(f_u_1, f_u_2, f_u_3, f_u_4)

        edge_map = self.eam(d, d1_u, d2_u)
        edge_feature = self.edge_feature(edge_map)

        end_sal = self.conv256_32(end_fuse_s)  # [b,32]
        end_sal1 = self.conv256_32(end_fuse_s1)
        up_edge = self.up_edge(edge_feature)

        out1 = self.relu(torch.cat((end_sal1, edge_feature), dim=1))
        out = self.relu(torch.cat((end_sal, edge_feature), dim=1))
        out = self.up4(out)
        out1 = self.up4(out1)
        sal_out = self.conv64_1(out)
        sal_out1 = self.conv64_1(out1)
        return sal_out, up_edge, sal_out1

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")


if __name__ == '__main__':
    net = Detector()
    a = torch.randn([2, 3, 384, 384])
    b = torch.randn([2, 3, 384, 384])
    s, e, s1 = net(a, b)
    print("s.shape:", e.shape)  # torch.Size([2, 1, 384, 384])


