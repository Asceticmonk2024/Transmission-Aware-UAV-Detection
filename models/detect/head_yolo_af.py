# models/detect/head_yolo_af.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

def _cba(c1, c2, k=3, s=1, p=None, g=1, act=True):
    if p is None:
        p = (k - 1) // 2
    return nn.Sequential(
        nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False),
        nn.BatchNorm2d(c2),
        nn.SiLU(inplace=True) if act else nn.Identity(),
    )

class _TwoConv(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = _cba(c, c, 3, 1)
        self.c2 = _cba(c, c, 3, 1)
    def forward(self, x):
        return self.c2(self.c1(x))

class DecoupleHeadAF(nn.Module):
    """
    简化 YOLO 风格 AF decoupled head：
      - 输入：多层 [B,C,S,S]
      - 输出：多层 [B, (4 + 1 + K), S, S]
    """
    def __init__(self, in_chs: List[int], num_classes: int = 3, hid: int = 128):
        super().__init__()
        self.num_classes = num_classes
        self.stems = nn.ModuleList([_cba(c, hid, 1, 1, p=0) for c in in_chs])
        self.cls_convs = nn.ModuleList([_TwoConv(hid) for _ in in_chs])
        self.reg_convs = nn.ModuleList([_TwoConv(hid) for _ in in_chs])
        self.cls_preds = nn.ModuleList([nn.Conv2d(hid, num_classes, 1) for _ in in_chs])
        self.obj_preds = nn.ModuleList([nn.Conv2d(hid, 1, 1) for _ in in_chs])
        self.box_preds = nn.ModuleList([nn.Conv2d(hid, 4, 1) for _ in in_chs])

    def forward(self, feats: List[torch.Tensor]):
        outs = []
        for i, f in enumerate(feats):
            x = self.stems[i](f)
            cls_f = self.cls_convs[i](x)
            reg_f = self.reg_convs[i](x)
            cls = self.cls_preds[i](cls_f)
            obj = self.obj_preds[i](reg_f)
            box = self.box_preds[i](reg_f)
            outs.append(torch.cat([box, obj, cls], dim=1))  # [B, 4+1+K, S, S]
        return outs
