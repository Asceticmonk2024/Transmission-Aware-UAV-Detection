#!/usr/bin/env python3
# tools/convert_coco_to_yolo.py
# 将 data/HazyDet/{train,val}/xxx_coco.json 转为 YOLOv8 标签：
# 输出到 data/HazyDet/{train,val}/labels/*.txt （与 hazy_images/*.jpg 同名）

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List
import argparse
import tqdm

# 固定类别映射（与你数据集一致）
# 你也可以自动从 JSON 读取 name->id 再排序，但为了稳定，这里固定：
NAME2IDX = {"car": 0, "truck": 1, "bus": 2}

def coco_to_yolo_box(x, y, w, h, img_w, img_h):
    # COCO: 左上角 xy + 宽高（绝对像素）
    # YOLO: 中心点 + 宽高（相对 0~1）
    cx = x + w / 2.0
    cy = y + h / 2.0
    return cx / img_w, cy / img_h, w / img_w, h / img_h

def convert_split(root: Path, split: str):
    ann_path = root / split / f"{split}_coco.json"
    img_dir  = root / split / "hazy_images"   # 你当前训练用 hazy_images
    out_dir  = root / split / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    assert ann_path.exists(), f"Not found: {ann_path}"
    assert img_dir.exists(),  f"Not found: {img_dir}"

    data = json.loads(ann_path.read_text())
    # 建索引
    imgs: Dict[int, dict] = {im["id"]: im for im in data.get("images", [])}
    # COCO category_id 可能不是 0..K-1，先根据 name 映射到连续 idx
    id2name: Dict[int, str] = {c["id"]: c["name"] for c in data.get("categories", [])}

    # 按 image_id 聚合标注
    img_to_anns: Dict[int, List[dict]] = {}
    for a in data.get("annotations", []):
        if a.get("iscrowd", 0) == 1:
            continue
        img_to_anns.setdefault(a["image_id"], []).append(a)

    n_img = 0
    n_lab = 0
    for img_id, info in tqdm.tqdm(imgs.items(), desc=f"Convert {split}"):
        file_name = Path(info["file_name"]).name
        stem = Path(file_name).with_suffix("").name
        W, H = int(info.get("width", 0)), int(info.get("height", 0))
        if W <= 0 or H <= 0:
            # 如果 JSON 里没存宽高，可以在这里用 OpenCV 读取一把（可选）
            # 但 HazyDet 的 JSON 一般有 width/height
            pass
        # 对应图像必须存在
        img_path = img_dir / file_name
        if not img_path.exists():
            # 跳过不存在的图片，避免写空标签误导 YOLO
            continue

        lines = []
        for ann in img_to_anns.get(img_id, []):
            cat_id = ann["category_id"]
            name = id2name.get(cat_id, None)
            if name not in NAME2IDX:
                continue
            cls_idx = NAME2IDX[name]
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            # 裁边（避免超出）
            x = max(0.0, min(float(x), W - 1.0))
            y = max(0.0, min(float(y), H - 1.0))
            w = max(1e-6, min(float(w), W - x))
            h = max(1e-6, min(float(h), H - y))
            cx, cy, ww, hh = coco_to_yolo_box(x, y, w, h, W, H)
            # 再次裁边（数值稳定）
            cx = min(max(cx, 0.0), 1.0)
            cy = min(max(cy, 0.0), 1.0)
            ww = min(max(ww, 1e-6), 1.0)
            hh = min(max(hh, 1e-6), 1.0)
            lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {ww:.6f} {hh:.6f}")

        out_txt = out_dir / f"{stem}.txt"
        if lines:
            out_txt.write_text("\n".join(lines) + "\n")
            n_lab += 1
        else:
            # 没有目标 → YOLO 允许**空文件**代表 background
            out_txt.write_text("")
        n_img += 1

    print(f"[{split}] images processed: {n_img}, non-empty labels: {n_lab}, out={out_dir}")

def main():
    ap = argparse.ArgumentParser("Convert HazyDet COCO to YOLO labels")
    ap.add_argument("--data", type=str, default="data/HazyDet")
    ap.add_argument("--splits", type=str, nargs="+", default=["train", "val"])
    args = ap.parse_args()

    root = Path(args.data)
    for sp in args.splits:
        convert_split(root, sp)

if __name__ == "__main__":
    main()
