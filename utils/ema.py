# utils/ema.py
import copy
import math
import torch

class ModelEMA:
    """
    轻量 EMA：在训练时调用 update(model)，保存/评估用 ema.ema
    """
    def __init__(self, model, decay=0.9998, device=None):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.device = device
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def _decay(self, i):
        # 按步数 warmup 一下
        return self.decay * (1 - math.exp(-i / 2000.0))

    @torch.no_grad()
    def update(self, model, step: int):
        d = self._decay(step)
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if k in msd:
                v.copy_(v * d + msd[k] * (1 - d))

    def to(self, device):
        self.ema.to(device)
        self.device = device
        return self
