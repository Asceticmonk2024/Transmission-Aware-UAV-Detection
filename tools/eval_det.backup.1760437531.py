#!/usr/bin/env python3
# tools/eval_det.py
import argparse, yaml, os, json, torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from math import ceil

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.hazydet_pair import HazyDetPair
from models.hfd_lite import HFDLite


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--ckpt",   type=str, required=True)
    ap.add_argument("--split",  type=str, default="val", choices=["val","train"])
    ap.add_argument("--conf",   type=float, default=0.25)
    ap.add_argument("--iou",    type=float, default=0.5)
    ap.add_argument("--max_det",type=int,   default=300)
    ap.add_argument("--device", type=str,   default="cuda")
    return ap.parse_args()


@torch.no_grad()
def box_iou(a, b):  # a:[M,4], b:[N,4] xyxy
    tl = torch.max(a[:, None, :2], b[:, :2])
    br = torch.min(a[:, None, 2:], b[:, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area_a[:, None] + area_b - inter
    return inter / (union + 1e-7)


@torch.no_grad()
def nms(boxes, scores, iou_th=0.5, max_det=300):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    keep = []
    idxs = scores.argsort(descending=True)
    while idxs.numel() > 0 and len(keep) < max_det:
        i = idxs[0]
        keep.append(i.item())
        if idxs.numel() == 1:
            break
        ious = box_iou(boxes[i].unsqueeze(0), boxes[idxs[1:]])[0]
        idxs = idxs[1:][ious <= iou_th]
    return torch.tensor(keep, device=boxes.device, dtype=torch.long)


@torch.no_grad()
def decode_levels_fcos_like(levels, img_hw, num_classes, strides=(8, 16, 32), conf_th=0.25):
    """
    levels: list of [B, 1+K+4, S, S], obj|cls[K]|box[4]
    采用 FCOS/YOLOv8-ltrb 风格解码：
      - obj, cls -> sigmoid
      - box -> relu，解释为距网格中心的 l,t,r,b 像素距离
      - 网格中心 = ((j+0.5)*stride, (i+0.5)*stride)
    返回：boxes[N,4] xyxy(像素), scores[N], labels[N]
    """
    B = levels[0].shape[0]
    assert B == 1, "此脚本按 batch=1 解码，如需批量评估请稍作改造。"
    H, W = img_hw

    all_boxes, all_scores, all_cls = [], [], []

    for lvl, stride in zip(levels, strides):
        # [1, 1+K+4, S, S]
        obj = torch.sigmoid(lvl[:, 0:1])          # [1,1,S,S]
        cls = torch.sigmoid(lvl[:, 1:1+num_classes])  # [1,K,S,S]
        box = F.relu(lvl[:, 1+num_classes:1+num_classes+4])  # [1,4,S,S]

        S = lvl.shape[-1]
        # 网格中心坐标（像素）
        # cx: [S,S], cy: [S,S]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(S, device=lvl.device),
            torch.arange(S, device=lvl.device),
            indexing="ij"
        )
        cx = (grid_x + 0.5) * stride
        cy = (grid_y + 0.5) * stride

        # 展平为 [N]
        obj = obj.reshape(-1)  # [S*S]
        cls = cls.reshape(num_classes, -1).transpose(0, 1)  # [S*S, K]
        scores, labels = cls.max(dim=1)                     # [S*S], [S*S]
        scores = scores * obj                               # 融合 obj

        # 置信度阈值
        keep = scores >= conf_th
        if keep.sum() == 0:
            continue

        # box: [1,4,S,S] -> [S*S,4]，ltrb 相对像素
        l = box[:, 0].reshape(-1)
        t = box[:, 1].reshape(-1)
        r = box[:, 2].reshape(-1)
        b = box[:, 3].reshape(-1)

        cxv = cx.reshape(-1)   # [S*S]
        cyv = cy.reshape(-1)

        x1 = (cxv - l).clamp(min=0, max=W - 1)
        y1 = (cyv - t).clamp(min=0, max=H - 1)
        x2 = (cxv + r).clamp(min=0, max=W - 1)
        y2 = (cyv + b).clamp(min=0, max=H - 1)

        all_boxes.append(torch.stack([x1, y1, x2, y2], dim=1)[keep])
        all_scores.append(scores[keep])
        all_cls.append(labels[keep])

    if len(all_boxes) == 0:
        return (torch.empty((0, 4), device=levels[0].device),
                torch.empty((0,), device=levels[0].device),
                torch.empty((0,), dtype=torch.long, device=levels[0].device))

    boxes = torch.cat(all_boxes, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_cls, dim=0)
    return boxes, scores, labels


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"

    # Dataset（你当前的 HazyDetPair 返回 {'hazy','clear','name'}）
    ds = HazyDetPair(cfg["data"]["root"], cfg["data"][f"split_{args.split}"],
                     cfg["data"]["image_size"], aug=False)

    # COCO GT
    ann_path = os.path.join(cfg["data"]["root"], cfg["data"][f"split_{args.split}"], f"{args.split}_coco.json")
    coco_gt = COCO(ann_path)

    # Model
    mcfg = cfg["model"]
    model = HFDLite(
        dehaze_channels=mcfg["dehaze_channels"],
        enable_detect=True,
        enable_cross=mcfg.get("enable_cross", True),
        num_classes=mcfg["num_classes"]
    ).to(device).eval()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    results = []  # COCO json
    for i in tqdm(range(len(ds)), ncols=100, desc="eval"):
        sample = ds[i]

        # 取 hazy 图像
        if isinstance(sample, dict) and "hazy" in sample:
            x = sample["hazy"].unsqueeze(0).to(device)  # [1,3,H,W]
            img_name = sample.get("name", f"img_{i:06d}")
        else:
            raise KeyError("HazyDetPair sample does not contain key 'hazy'.")

        B, _, H, W = x.shape
        out = model(x)

        # 优先使用模型已解码输出
        boxes = scores = labels = None
        if isinstance(out, dict) and "pred" in out:
            boxes  = out["pred"]["boxes"][0]
            scores = out["pred"]["scores"][0]
            labels = out["pred"]["labels"][0]
        else:
            # 外部解码：从 det_pred_levels 解码
            if not (isinstance(out, dict) and "det_pred_levels" in out):
                raise RuntimeError("Model forward did not return 'det_pred_levels' and no decoded 'pred'. "
                                   "Please ensure enable_detect=True during model build.")
            levels = out["det_pred_levels"]  # list of len=3
            boxes, scores, labels = decode_levels_fcos_like(
                levels, img_hw=(H, W),
                num_classes=mcfg["num_classes"],
                strides=(8, 16, 32),
                conf_th=args.conf
            )

        # 阈值内可能为空
        if boxes.numel():
            keep = nms(boxes, scores, iou_th=args.iou, max_det=args.max_det)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # to COCO json
        img_info = ds.coco_parser.img_name_to_info[Path(img_name).name]
        img_id = img_info["id"]

        # 反查原始 coco 类别 id（ds.coco_parser.cat_id_to_idx: coco_id -> contiguous_idx）
        inv_cat = {v: k for k, v in ds.coco_parser.cat_id_to_idx.items()}

        for b, sc, c in zip(boxes.cpu(), scores.cpu(), labels.cpu()):
            x1, y1, x2, y2 = b.tolist()
            w, h = x2 - x1, y2 - y1
            cat_id = inv_cat.get(int(c.item()), int(c.item()))
            results.append({
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [x1, y1, w, h],
                "score": float(sc.item())
            })

    # 保存与评估
    os.makedirs("tmp_eval", exist_ok=True)
    out_json = "tmp_eval/preds.json"
    with open(out_json, "w") as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(out_json)
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.evaluate(); E.accumulate(); E.summarize()


if __name__ == "__main__":
    main()
