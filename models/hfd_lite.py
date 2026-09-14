import torch, torch.nn as nn
from .dehaze.ig_two_branch_lite import IGTBDehazeLite
from .detect.decodet_lite_stub import DeCoDetLiteStub

class HFDLite(nn.Module):
    def __init__(self, dehaze_channels=64, enable_detect=False, enable_cross=False, num_classes=3):
        super().__init__()
        self.enable_detect = enable_detect
        self.enable_cross = enable_cross
        self.dehaze = IGTBDehazeLite(in_ch=3, base=dehaze_channels)
        self.detect = DeCoDetLiteStub(num_classes=num_classes) if enable_detect else None

    def forward(self, x, det_targets=None):
        clean, feats = self.dehaze(x)
        det_loss, det_out = {"loss_det": clean.sum()*0}, {}
        if self.enable_detect and self.detect is not None:
            det_loss, det_out = self.detect(feats, det_targets)
        return {"clean": clean, "feats": feats, "det_out": det_out, **det_loss}
