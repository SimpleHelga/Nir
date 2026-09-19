import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, resnet50
from typing import List, Tuple


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling."""
    def __init__(self, in_ch, out_ch, rates=[6, 12, 18]):
        super().__init__()
        self.stages = nn.ModuleList()
        self.stages.append(nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        ))
        for rate in rates:
            self.stages.append(ConvBNReLU(in_ch, out_ch, dilation=rate, padding=rate))
        self.stages.append(ConvBNReLU(in_ch, out_ch, kernel_size=1, padding=0))
        self.project = ConvBNReLU(out_ch * (len(rates) + 2), out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        h, w = x.shape[2:]
        res = []
        for stage in self.stages:
            y = stage(x)
            if y.shape[2:] != (h, w):
                y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)
            res.append(y)
        out = torch.cat(res, dim=1)
        return self.project(out)


class HierarchicalContextExtractor(nn.Module):
    """Multi-level feature fusion (M3-Net++)."""
    def __init__(self, in_chs: List[int], out_ch: int = 256):
        super().__init__()
        self.convs = nn.ModuleList()
        for ch in in_chs:
            self.convs.append(nn.Sequential(
                nn.Conv2d(ch, out_ch // len(in_chs), 1, bias=False),
                nn.BatchNorm2d(out_ch // len(in_chs)),
                nn.ReLU(inplace=True)
            ))
        self.fuse = ConvBNReLU(out_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]):
        outs = []
        for feat, conv in zip(features, self.convs):
            y = conv(feat)
            y = F.interpolate(y, size=target_size, mode='bilinear', align_corners=False)
            outs.append(y)
        return self.fuse(torch.cat(outs, dim=1))


class M3NetPlusPlus(nn.Module):
    """
    M3-Net++: Multi-scale, Multi-level, Multi-stream.
    """
    def __init__(self, num_classes=22, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            weights = 'IMAGENET1K_V2' if pretrained else None
            enc = resnet50(weights=weights)
            chs = [256, 512, 1024, 2048]
        else:
            weights = 'IMAGENET1K_V1' if pretrained else None
            enc = resnet34(weights=weights)
            chs = [64, 128, 256, 512]

        self.enc1 = nn.Sequential(enc.conv1, enc.bn1, enc.relu, enc.maxpool)
        self.enc2 = enc.layer1   # 1/4
        self.enc3 = enc.layer2   # 1/8
        self.enc4 = enc.layer3   # 1/16
        self.enc5 = enc.layer4   # 1/32

        self.aspp = ASPP(chs[3], 256)
        self.hce = HierarchicalContextExtractor([chs[0], chs[1], chs[2], 256], out_ch=256)

        # Увеличенный Dropout для борьбы с переобучением
        self.decoder = nn.Sequential(
            ConvBNReLU(256, 128), nn.Dropout(0.3),
            ConvBNReLU(128, 64), nn.Dropout(0.3),
        )
        self.detail_conv = nn.Sequential(ConvBNReLU(chs[0], 64, kernel_size=1, padding=0))
        self.final_fuse = ConvBNReLU(64 + 64, 64)
        self.seg_head = nn.Conv2d(64, num_classes, 1)
        self.aux_head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        input_size = x.shape[2:]
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        x5 = self.aspp(x5)
        fused = self.hce([x2, x3, x4, x5], target_size=x2.shape[2:])

        d = self.decoder(fused)
        d = F.interpolate(d, size=x2.shape[2:], mode='bilinear', align_corners=False)
        detail = self.detail_conv(x2)

        f = self.final_fuse(torch.cat([d, detail], dim=1))
        f = F.interpolate(f, size=input_size, mode='bilinear', align_corners=False)
        out = self.seg_head(f)

        aux = self.aux_head(fused)
        aux = F.interpolate(aux, size=input_size, mode='bilinear', align_corners=False)
        return {'out': out, 'aux': aux}


class HoVerNetAdapted(nn.Module):
    """
    HoVer-Net для семантической сегментации BCSS.
    3 головы: semantic (22 класса), hv_map (2 канала), boundary (1 канал).
    """
    def __init__(self, num_classes=22, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet50':
            weights = 'IMAGENET1K_V2' if pretrained else None
            enc = resnet50(weights=weights)
            chs = [256, 512, 1024, 2048]
        else:
            weights = 'IMAGENET1K_V1' if pretrained else None
            enc = resnet34(weights=weights)
            chs = [64, 128, 256, 512]

        self.enc1 = nn.Sequential(enc.conv1, enc.bn1, enc.relu, enc.maxpool)
        self.enc2 = enc.layer1
        self.enc3 = enc.layer2
        self.enc4 = enc.layer3
        self.enc5 = enc.layer4

        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(chs[3], 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(256 + chs[2], 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256 + chs[1], 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128 + chs[0], 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )

        self.semantic_head = nn.Sequential(ConvBNReLU(64, 64), nn.Conv2d(64, num_classes, 1))
        self.hv_head = nn.Sequential(ConvBNReLU(64, 64), nn.Conv2d(64, 2, 1))
        self.boundary_head = nn.Sequential(ConvBNReLU(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, x):
        input_size = x.shape[2:]
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        d5 = self.up5(x5)
        d4 = self.up4(torch.cat([d5, x4], dim=1))
        d3 = self.up3(torch.cat([d4, x3], dim=1))
        d2 = self.up2(torch.cat([d3, x2], dim=1))
        d2 = F.interpolate(d2, size=input_size, mode='bilinear', align_corners=False)

        return {
            'semantic': self.semantic_head(d2),
            'hv_map': self.hv_head(d2),
            'boundary': self.boundary_head(d2)
        }