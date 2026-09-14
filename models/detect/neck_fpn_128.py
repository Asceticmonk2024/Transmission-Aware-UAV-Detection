# models/detect/neck_fpn_128.py
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

def _cba(c1, c2, k=3, s=1, p=None, g=1, act=True):
    if p is None:
        p = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True) if act else nn.Identity(),
    )

class _TwoDWConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw1 = _cba(c, c, 3, 1, g=c)   # depthwise
        self.pw1 = _cba(c, c, 1, 1)
        self.dw2 = _cba(c, c, 3, 1, g=c)
        self.pw2 = _cba(c, c, 1, 1)
    def forward(self, x):
        return self.pw2(self.dw2(self.pw1(self.dw1(x))))

class LiteFPN128(nn.Module):
    """
    仅用到 s/4 与 s/8 两层特征，统一到 128 通道，构建 [P3(s/4), P4(s/8), P5(s/16)] 三层输出。
    参数名与你当前 HFDLiteDet 调用对齐：
      in_ch4=256, in_ch8=512, out_c=128
    """
    def __init__(self, in_ch4: int, in_ch8: int, out_c: int = 128):
        super().__init__()
        self.c3 = _cba(in_ch4, out_c, k=1, s=1, p=0)   # 1/4 -> 128
        self.c4 = _cba(in_ch8, out_c, k=1, s=1, p=0)   # 1/8 -> 128

        # top-down: p3 = cat(c3, up(c4)) -> fuse
        self.fuse_3 = _TwoDWConv(out_c * 2)

        # bottom-up: p4 = cat(down(p3), c4) -> fuse
        self.down4 = _cba(out_c * 2, out_c, k=3, s=2)  # stride 2
        self.fuse_4 = _TwoDWConv(out_c)

        # p5 = down(p4)
        self.down5 = _cba(out_c, out_c, k=3, s=2)
        self.refine5 = _TwoDWConv(out_c)

    def forward(self, f4: torch.Tensor, f8: torch.Tensor):
        # f4: (B, in_ch4, H/4, W/4)
        # f8: (B, in_ch8, H/8, W/8)
        p3_in = self.c3(f4)               # (B,128,H/4,W/4)
        p4_in = self.c4(f8)               # (B,128,H/8,W/8)

        up_p4 = F.interpolate(p4_in, scale_factor=2, mode='nearest')  # -> H/4
        p3 = self.fuse_3(torch.cat([p3_in, up_p4], 1))                # (B,256,H/4,W/4)

        p4 = self.down4(p3)                                          # (B,128,H/8,W/8)
        p4 = self.fuse_4(p4)                                         # (B,128,H/8,W/8)

        p5 = self.down5(p4)                                          # (B,128,H/16,W/16)
        p5 = self.refine5(p5)                                        # (B,128,H/16,W/16)

        # 返回 3 层给检测头
        return p3, p4, p5
