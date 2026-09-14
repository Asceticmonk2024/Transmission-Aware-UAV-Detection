import torch, torch.nn as nn

class DeCoDetLiteStub(nn.Module):
    """
    占位：后续我会把“深度条件卷积 + 轻量FPN + YOLO head”填充。
    现在返回 0 损失与空预测，便于先跑 Stage-A。
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, feats, targets=None):
        device = next(iter(feats.values())).device
        loss = torch.tensor(0.0, device=device)
        outputs = {}
        return {"loss_det": loss}, outputs
