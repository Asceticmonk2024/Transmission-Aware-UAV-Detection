# tools/assigners.py
import torch

def assign_iou_topk_center(boxes, anchors_xy, strides, gt_boxes, gt_classes,
                           iou_thr=0.5, topk=10, center_radius=2.5):
    """
    boxes: [N,4] 预测框 (输入图尺度)
    anchors_xy: [N,2] 网格中心(输入图尺度)
    strides: [N,1]
    gt_boxes: [M,4] (输入图尺度)
    gt_classes: [M]
    return: pos_idx [P], pos_gt_idx [P]
    """
    if gt_boxes.numel() == 0 or boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long), boxes.new_zeros((0,), dtype=torch.long)

    # 中心先验：以 gt 中心在各层取半径 = center_radius * stride 的圆/方窗内的候选
    gt_centers = 0.5*(gt_boxes[:, :2] + gt_boxes[:, 2:])
    # [N,M] 是否在中心窗内
    r = (center_radius * strides.squeeze(1)).unsqueeze(1)  # [N,1]
    in_center = (anchors_xy[:,0:1].between(gt_centers[:,0]-r, gt_centers[:,0]+r) &
                 anchors_xy[:,1:2].between(gt_centers[:,1]-r, gt_centers[:,1]+r))  # [N,M]

    # IoU 计算
    iou = box_iou(boxes, gt_boxes)  # [N,M]

    # 先筛中心窗，再按 IoU 阈值
    cand = in_center & (iou >= iou_thr)

    # 每个 gt 取 IoU Top-k
    topk_idx = []
    for j in range(gt_boxes.size(0)):
        cand_j = cand[:, j]
        if cand_j.any():
            iou_j = iou[cand_j, j]
            k = min(topk, iou_j.numel())
            vals, idx_local = torch.topk(iou_j, k, dim=0, largest=True)
            idx_global = cand_j.nonzero(as_tuple=False).squeeze(1)[idx_local]
            topk_idx.append(torch.stack([idx_global, torch.full_like(idx_global, j)], dim=1))
    if len(topk_idx) == 0:
        return boxes.new_zeros((0,), dtype=torch.long), boxes.new_zeros((0,), dtype=torch.long)

    pair = torch.cat(topk_idx, dim=0)  # [P,2]
    pos_idx, pos_gt = pair[:,0].long(), pair[:,1].long()

    # 解决“一框多 GT”：按 IoU 最大保留
    uniq, inv = torch.unique(pos_idx, return_inverse=True)
    best = torch.zeros_like(uniq)
    for k in range(uniq.numel()):
        cand = (inv == k).nonzero(as_tuple=False).squeeze(1)
        j = torch.argmax(iou[pos_idx[cand], pos_gt[cand]])
        best[k] = cand[j]
    pair_final = pair[best.long()]
    return pair_final[:,0], pair_final[:,1]

def box_iou(a, b):
    # a:[N,4] b:[M,4] xyxy
    tl = torch.max(a[:,None,:2], b[None,:, :2])
    br = torch.min(a[:,None,2:], b[None,:, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[...,0]*wh[...,1]
    area_a = ((a[:,2]-a[:,0])*(a[:,3]-a[:,1]))[:,None]
    area_b = ((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))[None,:]
    union = area_a + area_b - inter + 1e-9
    return inter / union
