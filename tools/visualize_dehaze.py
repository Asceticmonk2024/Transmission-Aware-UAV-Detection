#!/usr/bin/env python3
import os, argparse, yaml, torch
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

# 复用项目内模块
from datasets.hazydet_pair import HazyDetPair
from models.hfd_lite import HFDLite

# 与训练时保持一致的 SSIM/PSNR 实现（简单可靠）
import torch.nn.functional as F
def _ssim_map(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = F.avg_pool2d(x, 3, 1, 1); mu_y = F.avg_pool2d(y, 3, 1, 1)
    sigma_x = F.avg_pool2d(x*x,3,1,1) - mu_x**2
    sigma_y = F.avg_pool2d(y*y,3,1,1) - mu_y**2
    sigma_xy = F.avg_pool2d(x*y,3,1,1) - mu_x*mu_y
    ssim_n = (2*mu_x*mu_y + C1)*(2*sigma_xy + C2)
    ssim_d = (mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2)
    return torch.clamp(ssim_n/(ssim_d+1e-8), 0, 1)  # SSIM∈[0,1]

def calc_ssim(x, y):
    # x,y in [0,1], shape [B,3,H,W]
    return _ssim_map(x, y).mean().item()

def calc_psnr(x, y, eps=1e-10):
    # x,y in [0,1]
    mse = torch.mean((x - y) ** 2).item()
    if mse < eps: return 99.0
    import math
    return 10.0 * math.log10(1.0 / mse)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/hazydet_512.yaml",
                    help="训练时的配置，用来取图像尺寸/通道数")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Best ckpt 路径，如 ./checkpoints/hfd_lite/hfd_lite_hazydet_512_best.pt")
    ap.add_argument("--split", type=str, default="val", choices=["val","train"])
    ap.add_argument("--num", type=int, default=12, help="保存多少张样例")
    ap.add_argument("--outdir", type=str, default="samples_final")
    ap.add_argument("--batch", type=int, default=1, help="推理 batch（建议1以便文件名逐张保存）")
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 构造数据（和训练一致，禁用增广）
    ds = HazyDetPair(cfg["data"]["root"], split=args.split,
                     image_size=cfg["data"]["image_size"], aug=False)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)

    # 构造模型并加载权重（只用去雾分支即可，enable_detect False）
    model = HFDLite(dehaze_channels=cfg["model"]["dehaze_channels"],
                    enable_detect=False, enable_cross=False).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    miss, unexp = model.load_state_dict(ckpt["model"], strict=False)
    print(f"Loaded ckpt: {args.ckpt}\n  missing: {miss}\n  unexpected: {unexp}")
    model.eval()

    saved = 0
    ssim_list, psnr_list = [], []

    with torch.no_grad():
        for batch in dl:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)
            out = model(hazy)
            pred = out["clean"].clamp(0,1)

            # 评测
            ssim_val = calc_ssim(pred, clear)
            psnr_val = calc_psnr(pred, clear)
            ssim_list.append(ssim_val); psnr_list.append(psnr_val)

            # 逐张保存：hazy | pred | clear
            for b in range(hazy.size(0)):
                name = Path(batch["name"][b]).stem
                grid = make_grid(torch.stack([hazy[b], pred[b], clear[b]], dim=0), nrow=3)
                save_path = outdir / f"{saved:04d}_{name}.png"
                save_image(grid, save_path)
                saved += 1
                if saved >= args.num:
                    break
            if saved >= args.num:
                break

    import numpy as np
    print(f"\nSaved {saved} samples to: {outdir}")
    if len(ssim_list) > 0:
        print(f"Avg PSNR: {np.mean(psnr_list):.2f} dB  |  Avg SSIM: {np.mean(ssim_list):.4f}")

if __name__ == "__main__":
    main()
