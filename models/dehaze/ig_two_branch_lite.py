import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ConvBNReLU(nn.Sequential):
    def __init__(self, c1, c2, k=3, s=1, p=1):
        super().__init__(nn.Conv2d(c1,c2,k,s,p,bias=False), nn.BatchNorm2d(c2), nn.ReLU(inplace=True))

class LiteSelfAttn(nn.Module):
    """
    窗口化轻量自注意力：在 win×win 小块内做注意力，避免 HW×HW 显存爆炸。
    heads=2, win=4 更省显存。
    """
    def __init__(self, dim, heads=2, win=4):
        super().__init__()
        self.h = heads
        self.dim = dim
        self.win = win
        self.qkv = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)

    def forward(self, x):
        B, C, H, W = x.shape
        win = self.win

        # padding 到 win 的整数倍
        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = x.shape[-2:]

        # qkv
        q, k, v = self.qkv(x).chunk(3, dim=1)  # [B,C,Hp,Wp] ×3
        h = self.h
        c = C // h

        def reshape_windows(t):
            # [B,C,Hp,Wp] -> [B,h,c,Hp/win,win,Wp/win,win] -> [B,h,Nh,Nw,S,c]
            t = t.view(B, h, c, Hp // win, win, Wp // win, win)
            t = t.permute(0,1,3,5,4,6,2).contiguous().view(B, h, Hp//win, Wp//win, win*win, c)
            return t

        q_w = reshape_windows(q)                      # [B,h,Nh,Nw,S,c]
        k_w = reshape_windows(k).transpose(-1, -2)    # [B,h,Nh,Nw,c,S]
        v_w = reshape_windows(v)                      # [B,h,Nh,Nw,S,c]

        # 窗口内注意力
        scale = (c ** -0.5)
        attn = (q_w @ k_w) * scale                    # [B,h,Nh,Nw,S,S]
        attn = attn.softmax(dim=-1)
        out_w = attn @ v_w                            # [B,h,Nh,Nw,S,c]

        # 还原
        out_w = out_w.view(B, h, Hp//win, Wp//win, win, win, c)  # [B,h,Nh,Nw,win,win,c]
        out = out_w.permute(0,1,6,2,4,3,5).contiguous()          # [B,h,c,Nh,win,Nw,win]
        out = out.view(B, C, Hp, Wp)
        out = self.proj(out)

        # 去 padding
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out

class CrossGuidedBlock(nn.Module):
    """Transformer(全局) 引导 CNN(局部) 的交互块 + 轻量门控"""
    def __init__(self, c, heads=2, win=4):
        super().__init__()
        self.global_attn = LiteSelfAttn(c, heads=heads, win=win)
        self.local_conv  = ConvBNReLU(c, c)
        self.gate = nn.Sequential(nn.Conv2d(c*2, c, 1), nn.Sigmoid())

    def forward(self, x):
        g = self.global_attn(x)
        l = self.local_conv(x)
        w = self.gate(torch.cat([g,l], dim=1))
        return x + w * (g + l)

def _cp(module, x):
    # 封装 checkpoint 调用，节省显存
    return checkpoint(lambda inp: module(inp), x)

class IGTBDehazeLite(nn.Module):
    """Encoder-Decoder U-Net with CrossGuided blocks（轻量交互 + 检查点）"""
    def __init__(self, in_ch=3, base=48, use_ckpt=True):
        super().__init__()
        c = base
        self.use_ckpt = use_ckpt
        # Encoder
        self.enc1 = nn.Sequential(ConvBNReLU(in_ch,c), CrossGuidedBlock(c, heads=2, win=4))                  # 1/1
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(c,c*2), CrossGuidedBlock(c*2, 2, 4))           # 1/2
        self.enc3 = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(c*2,c*4), CrossGuidedBlock(c*4, 2, 4))         # 1/4
        # Bottleneck
        self.bott = nn.Sequential(nn.MaxPool2d(2), ConvBNReLU(c*4,c*8), CrossGuidedBlock(c*8, 2, 4))         # 1/8
        # Decoder
        self.up2 = nn.ConvTranspose2d(c*8, c*4, 2, 2)
        self.dec2 = nn.Sequential(ConvBNReLU(c*8, c*4), CrossGuidedBlock(c*4, 2, 4))
        self.up1 = nn.ConvTranspose2d(c*4, c*2, 2, 2)
        self.dec1 = nn.Sequential(ConvBNReLU(c*4, c*2), CrossGuidedBlock(c*2, 2, 4))
        self.up0 = nn.ConvTranspose2d(c*2, c, 2, 2)
        self.dec0 = nn.Sequential(ConvBNReLU(c*2, c), CrossGuidedBlock(c, 2, 4))
        # Head
        self.head = nn.Conv2d(c, 3, 1)

    def forward(self, x):
        if self.use_ckpt and self.training:
            e1 = _cp(self.enc1, x)                   # 1/1
            e2 = _cp(self.enc2, e1)                  # 1/2
            e3 = _cp(self.enc3, e2)                  # 1/4
            b  = _cp(self.bott, e3)                  # 1/8
            d2_in = torch.cat([self.up2(b), e3], dim=1); d2 = _cp(self.dec2, d2_in)
            d1_in = torch.cat([self.up1(d2), e2], dim=1); d1 = _cp(self.dec1, d1_in)
            d0_in = torch.cat([self.up0(d1), e1], dim=1); d0 = _cp(self.dec0, d0_in)
        else:
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            e3 = self.enc3(e2)
            b  = self.bott(e3)
            d2 = self.dec2(torch.cat([self.up2(b), e3], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1))
            d0 = self.dec0(torch.cat([self.up0(d1), e1], dim=1))

        out = torch.sigmoid(self.head(d0))
        feats = {"1/8": b, "1/4": e3}    # 供检测/FPN
        return out, feats
