# =============================================
# File: tools/inspect_head.py
# 打印每层 obj/cls 的 mean/max，判断检测头是否“活”
# 用法：PYTHONPATH=. python tools/inspect_head.py --ckpt checkpoints/xxx.pt
# =============================================
import argparse
import numpy as np
from PIL import Image
import torch
from pycocotools.coco import COCO


def main(args):
    from models.hfd_lite import HFDLite

    ann = "data/HazyDet/val/val_coco.json"
    img_dir = "data/HazyDet/val/hazy_images"

    coco = COCO(ann)
    info = coco.loadImgs([coco.getImgIds()[0]])[0]
    img = Image.open(f"{img_dir}/" + info["file_name"]).convert("RGB")
    arr = np.array(img.resize((512, 512))).astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    m = HFDLite().eval()
    raw = torch.load(args.ckpt, map_location="cpu")
    sd = raw.get("model", raw.get("state_dict", raw))
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)

    with torch.no_grad():
        out = m(x)
        preds = out[0] if (isinstance(out, tuple) and len(out) >= 1) else (out if isinstance(out, (list, tuple)) else [out])

    for i, p in enumerate(preds):
        t = p.permute(0, 2, 3, 1)
        obj = t[..., 4].sigmoid()
        cls = t[..., 5:].sigmoid().max(-1).values
        print(f"layer{i}: shape={tuple(p.shape)}, obj(mean,max)=({obj.mean():.4f},{obj.max():.4f}), cls(mean,max)=({cls.mean():.4f},{cls.max():.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    args = ap.parse_args()
    main(args)
