# models/hfd_lite_det_official.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.dehaze.ig_two_branch_lite import IGTBDehazeLite
from models.detect.neck_pafpn_lite import LitePAFPN  # 你现有的轻量 FPN，若文件名不同改这里
from models.detect.yolo8_official import Y8OfficialHeadWrapper


class HFDLiteDetOfficial(nn.Module):
    """
    去雾 backbone + 轻量 FPN + 官方 YOLOv8 检测头（Detect）
    forward 推理时返回 dets [N,6]（直接 decode 完成）
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.nc = num_classes
        self.dehaze = IGTBDehazeLite(in_ch=3, base=64)

        # 假定 dehaze 吐出 feats: dict{'1/4':(B,256,H/4,W/4), '1/8':(B,512,H/8,W/8)} 或类似
        # 这里 neck 接两个输入，内部补一个 P5。若你的 neck 构造不同，请改构造函数和 forward 对应对接。
        self.neck = LitePAFPN(ch_in=(256, 512, 512), ch_out=128)  # (C3,C4,C5) -> 每层到 128
        self.head = Y8OfficialHeadWrapper(num_classes=num_classes, ch_in=(128, 128, 128))

    def _collect_backbone_feats(self, feats_any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从 dehaze 的 feats 中取出 3 层用于检测：
        - 先取 1/8, 1/16, 1/32；若只有 1/4 和 1/8，就把 1/8 下采样一次作为 1/16，再次下采样作为 1/32
        """
        if isinstance(feats_any, dict):
            f4 = feats_any.get("1/4", None)
            f8 = feats_any.get("1/8", None)
            f16 = feats_any.get("1/16", None)
        else:
            raise RuntimeError("dehaze feats 应该是 dict，包含 '1/4','1/8','1/16' 之类的键")

        if f16 is None and f8 is not None:
            f16 = F.avg_pool2d(f8, 2, 2)
        if f4 is None or f8 is None or f16 is None:
            raise RuntimeError("缺少必要的特征层（至少需要 1/4 与 1/8，另外 1/16 可由 1/8 下采样得到）")

        return f4, f8, f16

    def forward(self, x: torch.Tensor, *, infer: bool = True,
                conf: float = 0.25, iou: float = 0.50, max_det: int = 300):
        """
        infer=True：返回 dets [M,6]。
        （如果你后面要训练，另写 train_step，在那里面调用 self.head.detect(out_feats) 的 raw logits 走官方损失。）
        """
        with torch.set_grad_enabled(not infer):
            clean, feats = self.dehaze(x)   # clean: 去雾图；feats: 多尺度特征
            f4, f8, f16 = self._collect_backbone_feats(feats)
            p3, p4, p5 = self.neck(f4, f8, f16)  # 输出三层统一到 128 通道

            if infer:
                dets = self.head.forward_infer([p3, p4, p5], img_size=(x.shape[-1], x.shape[-2]),
                                               conf_thres=conf, iou_thres=iou, max_det=max_det)
                return dets
            else:
                # 训练时可返回 raw for loss；这里简化直接把 feats 传出去
                return [p3, p4, p5]
