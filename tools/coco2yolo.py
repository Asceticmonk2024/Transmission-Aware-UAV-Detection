#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 COCO JSON (xywh, 左上角为原点, 绝对像素) 转为 YOLO txt (class cx cy w h, 归一化到[0,1])。
- 会自动读取 images、annotations、categories
- 跳过 iscrowd=1、跳过面积<=0的框、并裁剪到图像边界
- labels 输出到 train/labels 与 val/labels
- 类别顺序按 names_order 硬编码为 ['car','truck','bus']；若你的 COCO 里类别名不同，改这个列表即可
用法:
  PYTHONPATH=. python tools/coco2yolo.py \
    --root data/HazyDet \
    --train-json data/HazyDet/train/train_coco.json \
    --val-json   data/HazyDet/val/val_coco.json
"""
import argparse, json
from pathlib import Path

def coco_to_yolo(coco_json_path: Path, img_dir: Path, labels_dir: Path, names_order):
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    # COCO categories -> 基于名字到索引（保持你要的顺序）
    # names_order 决定最终的 class id：car->0, truck->1, bus->2
    name2yid = {n: i for i, n in enumerate(names_order)}

    # COCO 本身的类别 id 映射到 yid（通过名字）
    cid2yid = {}
    for c in coco.get("categories", []):
        n = c["name"]
        if n not in name2yid:
            # 允许忽略不在 names_order 里的类别
            continue
        cid2yid[c["id"]] = name2yid[n]

    # images 索引
    imgid2info = {im["id"]: im for im in coco.get("images", [])}

    # 收集每张图的标注
    per_image_lines = {}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd", 0) == 1:
            continue
        img_id = ann["image_id"]
        if img_id not in imgid2info:
            continue
        im = imgid2info[img_id]
        W, H = im["width"], im["height"]
        if W <= 0 or H <= 0:
            continue

        cid = ann["category_id"]
        if cid not in cid2yid:
            # 不是 car/truck/bus，忽略
            continue
        yid = cid2yid[cid]

        x, y, w, h = ann["bbox"]  # COCO 是 xywh, 绝对像素
        # 裁剪到图像范围
        x1 = max(0.0, min(float(x), W - 1.0))
        y1 = max(0.0, min(float(y), H - 1.0))
        x2 = max(0.0, min(float(x + w), W - 1.0))
        y2 = max(0.0, min(float(y + h), H - 1.0))
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 1e-6 or bh <= 1e-6:
            continue

        cx = x1 + bw * 0.5
        cy = y1 + bh * 0.5
        # 归一化
        cxn = cx / W
        cyn = cy / H
        bwn = bw / W
        bhn = bh / H

        line = f"{yid} {cxn:.6f} {cyn:.6f} {bwn:.6f} {bhn:.6f}"
        fname = Path(im["file_name"]).stem + ".txt"
        per_image_lines.setdefault(fname, []).append(line)

    # 写出
    n_img = 0
    n_box = 0
    for txt_name, lines in per_image_lines.items():
        # 对应图像实际是否存在
        # YOLOv8 会按 images 目录中的文件名去找同名 .txt
        # 这里只写 txt；是否存在同名 jpg/png 由数据本身决定
        (labels_dir / txt_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
        n_img += 1
        n_box += len(lines)

    print(f"[coco2yolo] {coco_json_path.name}: 写出 {n_img} 张的 labels，共 {n_box} 个框 -> {labels_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="数据根目录（包含 train/ 和 val/）")
    ap.add_argument("--train-json", type=str, required=True)
    ap.add_argument("--val-json", type=str, required=True)
    ap.add_argument("--names", type=str, default="car,truck,bus", help="类别名顺序，逗号分隔")
    args = ap.parse_args()

    names_order = [s.strip() for s in args.names.split(",") if s.strip()]
    root = Path(args.root)
    coco_to_yolo(Path(args.train_json), root/"train"/"hazy_images", root/"train"/"labels", names_order)
    coco_to_yolo(Path(args.val_json)  , root/"val"/"hazy_images",   root/"val"/"labels",   names_order)

if __name__ == "__main__":
    main()
