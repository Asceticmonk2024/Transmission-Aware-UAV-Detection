# models/detect/yolo8_official.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 官方 YOLOv8 检测头与解码工具
from ultralytics.nn.modules.head import Detect   # 官方 Head
from ultralytics.utils.ops import make_anchors, dist2bbox, non_max_suppression


class Y8OfficialHeadWrapper(nn.Module):
    """
    一个薄包装：把你提供的 P3/P4/P5 特征（通道随意）交给官方 YOLOv8 Detect
    再用官方 ops 解码，给出最终 dets[N,6] (xyxy, conf, cls)。

    注意：
    - 输入图像我们统一 resize 到 512×512 推理；strides 由特征图尺寸自动推断。
    - ch_in 会被 Detect 接受；Detect 内部自己用 1×1 做通道适配，所以你不用手动对齐通道。
    """

    def __init__(self, num_classes: int = 3, ch_in: List[int] | Tuple[int, int, int] = (128, 128, 128)):
        super().__init__()
        self.nc = int(num_classes)
        self.detect = Detect(nc=self.nc, ch=ch_in)  # 这就是官方头

        # 训练/推理都需要用到 anchor grid（官方解码走的距离分布）
        self.register_buffer("_anc_cache_grid", torch.zeros(1))

    def _infer_strides(self, feats: List[torch.Tensor]) -> List[int]:
        # 输入已经被外部统一到 512×512；根据各层 W 推断 stride
        s = []
        for f in feats:
            s.append(int(round(512 / f.shape[-1])))
        return s  # e.g. [8,16,32]

    @torch.no_grad()
    def forward_infer(self, feats: List[torch.Tensor],
                      img_size: Tuple[int, int] = (512, 512),
                      conf_thres: float = 0.25, iou_thres: float = 0.50,
                      max_det: int = 300) -> torch.Tensor:
        """
        推理（无梯度）：给出 dets [N,6] (x1,y1,x2,y2, conf, cls)
        """
        device = feats[0].device
        B = feats[0].shape[0]
        assert B == 1, "当前包装只在推理时支持 batch=1（评测就是按图跑）。"

        # 官方 Detect 前向：返回分层 raw logits（DFL 需要）
        # x: list[Tensors]，与传入层级同
        raw = self.detect(feats)  # raw 是 list，每层 [B, no, H, W]
        strides = self._infer_strides(feats)

        # 官方 anchor 生成（每层）
        anchors, grid_strides = make_anchors(feats, strides, 0.5, device=device)  # (sum(HW),2), (sum(HW),1)
        # 官方解码：raw[..., :4] 是距离分布（经过 Detect 内部 DFL conv），要先做 softmax→期望，再 dist2bbox
        # Detect 已经把 DFL 的 conv 加在 head 里了，这里 raw 是直接回归量
        # Ultralytics 内部的常规路径是：pred = self.detect(feats); 再 self.detect.postprocess(...)
        # 我们参考其 ops：先 reshape 再合并层，再 dist2bbox

        # 合层
        pred_list = []
        for r in raw:  # [B, no, H, W]
            b, c, h, w = r.shape
            pred_list.append(r.view(b, c, -1))  # [B, C, HW]
        pred = torch.cat(pred_list, dim=2)       # [B, C, sum(HW)]
        pred = pred.permute(0, 2, 1).contiguous()  # [B, sum(HW), C]
        # C = 4(分布已回归为距离) + nc（分类）  （官方 v8 的 Detect 是 decoupled + DFL）

        # 取回 box、cls
        box_raw = pred[..., 0:4]         # [B,N,4] 是 (l,t,r,b) 的距离
        cls_logit = pred[..., 4:]        # [B,N,nc]

        # 距离转为 xyxy（以 grid 为基准）
        # anchors: [N,2] 是 grid 中心点；grid_strides: [N,1]
        boxes = dist2bbox(box_raw, anchors, grid_strides, xywh=False)  # [B,N,4] -> xyxy

        # 分数
        scores, labels = cls_logit.sigmoid().max(dim=-1)   # [B,N]
        # NMS 前拼接
        dets_xyxy = boxes[0]                               # [N,4]
        dets = torch.cat([dets_xyxy, scores[0:1].transpose(0,1)], dim=1)  # shape 占位，下面直接用 nms 函数

        # 官方 NMS（注意输入格式：列表形式 [ (xyxy, conf, cls), ... ] ）
        # 我们构造符合接口的 tensor: [N, 6] -> xyxy + conf + clsid
        dets_yolo = torch.cat([
            dets_xyxy,                      # [N,4]
            scores[0].unsqueeze(1),         # [N,1]
            labels[0].float().unsqueeze(1)  # [N,1]
        ], dim=1)

        # 交给官方 nms
        out = non_max_suppression(
            dets_yolo.unsqueeze(0),         # [1,N,6]
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            classes=None,
            agnostic=False,
            multi_label=False,
            max_det=max_det
        )[0]  # [M,6]
        return out  # xyxy, conf, cls
