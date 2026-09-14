# tools/decoders.py
import torch
import torch.nn.functional as F

@torch.no_grad()
def build_grid_and_stride(feat_shapes, strides, device, dtype):
    # feat_shapes: [(h1,w1), (h2,w2), (h3,w3)]
    # strides: [8, 16, 32] 等
    grids, stride_tensors = [], []
    for (h, w), s in zip(feat_shapes, strides):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        grid = torch.stack((x, y), dim=-1).view(-1, 2).to(dtype)  # [HW, 2], (cx, cy) in grid coords
        grids.append(grid)
        stride_tensors.append(torch.full((h*w, 1), float(s), device=device, dtype=dtype))
    return torch.cat(grids, 0), torch.cat(stride_tensors, 0)  # [N,2], [N,1]

def dfl_project(dist, reg_max=16):
    # dist: [..., reg_max]  logits for each bin
    # returns expected distance in bins
    p = F.softmax(dist, dim=-1)
    bins = torch.arange(reg_max, device=dist.device, dtype=dist.dtype).view(1, -1)
    return (p * bins).sum(dim=-1)

def decode_yolo_dfl(pred, feat_shapes, strides, num_classes, reg_max=16, img_size=640):
    """
    pred: list of [B, (4*reg_max + C), H, W] from each level OR already cat to [B, sum(HW), ...]
    Return: boxes_xyxy [B, N, 4] in input image scale, scores [B, N, C]
    """
    device, dtype = pred[0].device, pred[0].dtype
    # concat levels -> [B, HW, 4*reg_max + C]
    outs = []
    for p in pred:
        B, ch, H, W = p.shape
        outs.append(p.permute(0, 2, 3, 1).reshape(B, H*W, ch))
    out = torch.cat(outs, dim=1)
    B, N, ch = out.shape
    reg = out[..., :4*reg_max].reshape(B, N, 4, reg_max)
    cls = out[..., 4*reg_max:]  # [B, N, C]

    # grids & strides
    grid, stride_t = build_grid_and_stride(feat_shapes, strides, device, dtype)  # [N,2], [N,1]
    grid = grid.unsqueeze(0).expand(B, -1, -1)  # [B,N,2]
    stride_t = stride_t.unsqueeze(0).expand(B, -1, -1)  # [B,N,1]

    # DFL distances in bins -> expected -> pixels on feature map
    ltrb = dfl_project(reg, reg_max=reg_max)  # [B,N,4] distances in bins
    # 将“bin单位距离”映射为像素：每个 bin 对应 1 个格点单位，再乘 stride 得到输入图尺度
    ltrb = ltrb * stride_t  # [B,N,4] in input-pixel

    # center (cx,cy) 在输入图尺度： (grid + 0.5) * stride
    centers = (grid + 0.5) * stride_t  # [B,N,2]
    cx, cy = centers[..., 0], centers[..., 1]
    l, t, r, b = ltrb.unbind(-1)
    x1 = (cx - l).clamp(0, img_size - 1)
    y1 = (cy - t).clamp(0, img_size - 1)
    x2 = (cx + r).clamp(0, img_size - 1)
    y2 = (cy + b).clamp(0, img_size - 1)
    boxes = torch.stack([x1, y1, x2, y2], dim=-1)  # [B,N,4] xyxy

    scores = cls.sigmoid()  # [B,N,C]
    return boxes, scores
