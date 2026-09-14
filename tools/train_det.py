# tools/train_det.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, json, math, time, argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from tqdm import tqdm

from models.hfd_lite_det import HFDLiteDet

# ---------------- Dataset ----------------
class HazyDetCocoDataset(Dataset):
    def __init__(self, root: str, split: str = "train", img_size: int = 512):
        self.root = Path(root)
        self.split = split
        ann = self.root / split / (f"{split}_coco.json")
        img_dir = self.root / split / "hazy_images"
        print(f"[Dataset] split={split}  ann={ann}  imgs={img_dir}")
        self.coco = COCO(str(ann))
        self.img_dir = img_dir
        self.ids = self.coco.getImgIds()
        self.img_size = img_size

    def __len__(self): return len(self.ids)

    def _load_img(self, info):
        p = self.img_dir / info["file_name"]
        im = Image.open(p).convert("RGB").resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(im).astype(np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        return x

    def _load_targets(self, img_id, hw_orig: Tuple[int,int]) -> Dict[str, torch.Tensor]:
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=[img_id]))
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1e-3 or h <= 1e-3:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(int(a["category_id"]))
        if len(boxes) == 0:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        return {"boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.long)}

    def __getitem__(self, i: int):
        img_id = self.ids[i]
        info = self.coco.loadImgs([img_id])[0]
        x = self._load_img(info)
        t = self._load_targets(img_id, (info["height"], info["width"]))
        return x, t

def collate_fn(batch):
    xs, ts = zip(*batch)
    x = torch.stack(xs, dim=0)
    return x, list(ts)

# ---------------- Loss ----------------
@torch.no_grad()
def build_grid_strides(preds: List[torch.Tensor], strides: List[float], img_size: int):
    device = preds[0].device
    outs = []
    for p, s in zip(preds, strides):
        _, _, h, w = p.shape
        ys, xs = torch.meshgrid(torch.arange(h, device=device),
                                torch.arange(w, device=device),
                                indexing='ij')
        grid_xy = torch.stack([xs + 0.5, ys + 0.5], dim=-1).float() * float(s)  # [H,W,2]
        outs.append((grid_xy, float(s)))
    return outs

class SimpleYOLOLoss(nn.Module):
    def __init__(self, num_classes: int = 3, img_size: int = 512):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, preds: List[torch.Tensor], strides: List[float], targets: List[Dict[str, torch.Tensor]]):
        device = preds[0].device
        B = preds[0].shape[0]
        K = self.num_classes
        grid_info = build_grid_strides(preds, strides, self.img_size)

        total_obj, total_cls, total_box = 0.0, 0.0, 0.0
        total_pos = 0

        for b in range(B):
            all_centers, all_level_refs = [], []
            for li, (p, (grid_xy, s)) in enumerate(zip(preds, grid_info)):
                _, C, H, W = p.shape
                all_centers.append(grid_xy.view(-1, 2))
                hh, ww = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device), indexing='ij')
                ref = torch.stack([torch.full_like(hh, li), hh, ww], dim=-1).view(-1, 3)
                all_level_refs.append(ref)
            all_centers = torch.cat(all_centers, 0)
            all_level_refs = torch.cat(all_level_refs, 0)

            gt = targets[b]
            gt_boxes = gt["boxes"].to(device)
            gt_labels = gt["labels"].to(device)

            if gt_boxes.numel() == 0:
                for li, p in enumerate(preds):
                    obj_logit = p[b, 0]
                    total_obj += self.bce(obj_logit, torch.zeros_like(obj_logit)).mean()
                continue

            gt_centers = 0.5 * (gt_boxes[:, 0:2] + gt_boxes[:, 2:4])
            d2 = (all_centers[:, None, :] - gt_centers[None, :, :]).pow(2).sum(-1)
            pos_idx = d2.argmin(dim=0)  # [Ng]

            for gi in range(gt_boxes.shape[0]):
                li, hh, ww = all_level_refs[pos_idx[gi]].tolist()
                p = preds[li]
                raw = p[b]  # [C,H,W]
                obj_logit = raw[0]
                cls_logit = raw[1:1 + K]
                box_raw  = raw[1 + K: 1 + K + 4]

                obj_t = torch.zeros_like(obj_logit); obj_t[hh, ww] = 1.0
                total_obj += self.bce(obj_logit, obj_t).mean()

                cls_t = torch.zeros_like(cls_logit); cls_t[gt_labels[gi], hh, ww] = 1.0
                total_cls += self.bce(cls_logit, cls_t).mean()

                x1, y1, x2, y2 = gt_boxes[gi]
                gx = 0.5 * (x1 + x2); gy = 0.5 * (y1 + y2)
                gw = torch.clamp(x2 - x1, min=1.0); gh = torch.clamp(y2 - y1, min=1.0)
                s = float(strides[li])
                cx_t = torch.clamp(gx / s - ww, -0.5, 1.5)
                cy_t = torch.clamp(gy / s - hh, -0.5, 1.5)
                w_t = torch.sqrt(torch.clamp(gw / s, min=1e-3))
                h_t = torch.sqrt(torch.clamp(gh / s, min=1e-3))
                box_t = torch.stack([cx_t, cy_t, w_t, h_t], dim=0)
                total_box += self.smoothl1(box_raw[:, hh, ww], box_t).mean()
                total_pos += 1

        denom = max(total_pos, 1)
        loss_obj = total_obj / denom
        loss_cls = total_cls / denom
        loss_box = total_box / denom
        loss = loss_obj + loss_cls + loss_box
        return {"loss": loss, "loss_obj": loss_obj, "loss_cls": loss_cls, "loss_box": loss_box}

# ---------------- Checkpoint ----------------
class CheckpointManager:
    def __init__(self, out_dir: Path, keep_last: int = 5):
        self.weights = out_dir / "weights"
        self.weights.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.saved_epochs: List[Path] = []

    def _pack(self, model: nn.Module, optimizer, scaler, epoch: int, step: int, args, best_val: float | None):
        return {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "step": step,
            "best_val": best_val,
            "args": vars(args),
            "time": time.asctime(),
        }

    def save_last(self, model, optimizer, scaler, epoch, step, args, best_val):
        torch.save(self._pack(model, optimizer, scaler, epoch, step, args, best_val), self.weights / "last.pt")

    def save_best(self, model, optimizer, scaler, epoch, step, args, best_val):
        torch.save(self._pack(model, optimizer, scaler, epoch, step, args, best_val), self.weights / "best.pt")

    def save_epoch(self, model, optimizer, scaler, epoch, step, args, best_val):
        p = self.weights / f"epoch_{epoch:04d}.pt"
        torch.save(self._pack(model, optimizer, scaler, epoch, step, args, best_val), p)
        self.saved_epochs.append(p)
        if self.keep_last > 0 and len(self.saved_epochs) > self.keep_last:
            old = self.saved_epochs.pop(0)
            try: old.unlink()
            except Exception: pass

    @staticmethod
    def load_for_resume(model: nn.Module, optimizer, scaler, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ck["model"], strict=False)
        if optimizer is not None and ck.get("optimizer") is not None:
            optimizer.load_state_dict(ck["optimizer"])
        if scaler is not None and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        return ck.get("epoch", 0), ck.get("step", 0), ck.get("best_val", None)

# ---------------- Eval ----------------
@torch.no_grad()
def evaluate(model: nn.Module, loss_fn: nn.Module, loader, device):
    model.eval()
    tot, n = 0.0, 0
    for x, t in loader:
        x = x.to(device, non_blocking=True)
        preds, strides = model(x)
        losses = loss_fn(preds, strides, t)
        tot += float(losses["loss"].item())
        n += 1
    model.train()
    return tot / max(n, 1)

# ---------------- Train ----------------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = HazyDetCocoDataset(args.data, split="train", img_size=args.img_size)
    val_set   = HazyDetCocoDataset(args.data, split="val",   img_size=args.img_size)
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=4,
                              pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=2,
                              pin_memory=True, collate_fn=collate_fn)

    model = HFDLiteDet(num_classes=3).to(device)
    print("[model] fresh HFDLiteDet (no checkpoint loaded).")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    loss_fn = SimpleYOLOLoss(num_classes=3, img_size=args.img_size)

    start_epoch, global_step, best_val = 1, 0, None
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    saver = CheckpointManager(out_dir, keep_last=args.keep_last)

    if args.resume:
        print(f"[resume] loading from {args.resume}")
        se, gs, bv = CheckpointManager.load_for_resume(model, optimizer, scaler, args.resume)
        start_epoch = int(se) + 1; global_step = int(gs); best_val = bv
        print(f"[resume] start_epoch={start_epoch} best_val={best_val}")

    total_epochs = int(args.epochs_frozen + args.epochs_joint)
    if hasattr(model, "dehaze"):
        for p in model.dehaze.parameters(): p.requires_grad_(False)

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{total_epochs}")
        for x, t in pbar:
            x = x.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                preds, strides = model(x)
                losses = loss_fn(preds, strides, t)
                loss = losses["loss"]

            # ---- FIXED: 正确的 AMP 顺序 ----
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # --------------------------------

            global_step += 1
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "obj": f"{losses['loss_obj'].item():.3f}",
                "cls": f"{losses['loss_cls'].item():.3f}",
                "box": f"{losses['loss_box'].item():.3f}",
            })

        val_loss = evaluate(model, loss_fn, val_loader, device)
        if best_val is None or val_loss < best_val:
            best_val = val_loss
            saver.save_best(model, optimizer, scaler, epoch, global_step, args, best_val)
        if args.save_every and (epoch % args.save_every == 0):
            saver.save_epoch(model, optimizer, scaler, epoch, global_step, args, best_val)
        saver.save_last(model, optimizer, scaler, epoch, global_step, args, best_val)

        if epoch == args.epochs_frozen and hasattr(model, "dehaze"):
            for p in model.dehaze.parameters(): p.requires_grad_(True)
            for g in optimizer.param_groups: g["lr"] = args.lr2
            print(f"[Stage2] unfreeze dehaze, set lr={args.lr2}")

    print(f"[done] weights saved to: {out_dir/'weights'}")

# ---------------- Args ----------------
def build_parser():
    p = argparse.ArgumentParser("HFDLiteDet training with robust checkpointing")
    p.add_argument("--data", type=str, default="data/HazyDet")
    p.add_argument("--img-size", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs-frozen", type=int, default=10)
    p.add_argument("--epochs-joint", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr2", type=float, default=5e-4)
    p.add_argument("--amp", action="store_true", help="enable mixed precision (CUDA only)")
    p.add_argument("--out", type=str, default="runs/train_det_simplified")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--keep-last", type=int, default=5)
    p.add_argument("--resume", type=str, default="")
    return p

if __name__ == "__main__":
    args = build_parser().parse_args()
    train(args)
