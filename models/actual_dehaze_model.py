# models/actual_dehaze_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActualDehazeModel(nn.Module):
    """基于你实际训练权重的去雾模型结构"""
    
    def __init__(self, base=48):
        super().__init__()
        
        # 根据实际权重分析的精确结构：
        # enc1: [48,3,3,3] 
        # down1: [96,48,3,3]
        # enc2: [96,96,3,3]
        # down2: [192,96,3,3] 
        # enc3: [192,192,3,3]
        # up2: [192,96,2,2] - 注意这里是192->96
        # dec2: [96,192,3,3] - 输入是concat后的192
        # up1: [96,48,2,2]
        # dec1: [48,96,3,3] - 输入是concat后的96
        # out: [3,48,1,1]
        
        # 编码器路径
        self.enc1 = nn.Conv2d(3, 48, 3, 1, 1)
        self.enc1_bn = nn.BatchNorm2d(48)
        
        self.down1 = nn.Conv2d(48, 96, 3, 1, 1)  
        self.down1_bn = nn.BatchNorm2d(96)
        
        self.enc2 = nn.Conv2d(96, 96, 3, 1, 1)
        self.enc2_bn = nn.BatchNorm2d(96)
        
        self.down2 = nn.Conv2d(96, 192, 3, 1, 1)
        self.down2_bn = nn.BatchNorm2d(192)
        
        self.enc3 = nn.Conv2d(192, 192, 3, 1, 1)
        self.enc3_bn = nn.BatchNorm2d(192)
        
        # 解码器路径 - 关键修复
        self.up2 = nn.ConvTranspose2d(192, 96, 2, 2)  # 192->96 匹配权重
        self.dec2 = nn.Conv2d(192, 96, 3, 1, 1)       # 96(up)+96(enc2)=192 -> 96
        self.dec2_bn = nn.BatchNorm2d(96)
        
        self.up1 = nn.ConvTranspose2d(96, 48, 2, 2)   # 96->48
        self.dec1 = nn.Conv2d(96, 48, 3, 1, 1)        # 48(up)+48(enc1)=96 -> 48  
        self.dec1_bn = nn.BatchNorm2d(48)
        
        # 输出层
        self.out = nn.Conv2d(48, 3, 1, 1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        print(f"实际去雾模型初始化完成 (base={base})")
    
    def forward(self, x):
        # 编码路径
        e1 = F.relu(self.enc1_bn(self.enc1(x)))          # [B, 48, H, W]
        
        d1 = F.relu(self.down1_bn(self.down1(e1)))       # [B, 96, H, W]
        p1 = self.pool(d1)                               # [B, 96, H/2, W/2]
        
        e2 = F.relu(self.enc2_bn(self.enc2(p1)))         # [B, 96, H/2, W/2]
        
        d2 = F.relu(self.down2_bn(self.down2(e2)))       # [B, 192, H/2, W/2]
        p2 = self.pool(d2)                               # [B, 192, H/4, W/4]
        
        e3 = F.relu(self.enc3_bn(self.enc3(p2)))         # [B, 192, H/4, W/4]
        
        # 解码路径
        u2 = self.up2(e3)                               # [B, 96, H/2, W/2]
        concat2 = torch.cat([u2, e2], dim=1)            # [B, 192, H/2, W/2]
        dec2 = F.relu(self.dec2_bn(self.dec2(concat2))) # [B, 96, H/2, W/2]
        
        u1 = self.up1(dec2)                             # [B, 48, H, W]
        concat1 = torch.cat([u1, e1], dim=1)            # [B, 96, H, W]  
        dec1 = F.relu(self.dec1_bn(self.dec1(concat1))) # [B, 48, H, W]
        
        # 输出
        out = torch.sigmoid(self.out(dec1))              # [B, 3, H, W]
        
        return out

def test_actual_model():
    """测试模型结构是否正确"""
    
    model = ActualDehazeModel()
    
    # 测试前向传播
    x = torch.randn(1, 3, 512, 512)
    
    try:
        with torch.no_grad():
            out = model(x)
        print(f"模型测试成功: 输入 {x.shape} -> 输出 {out.shape}")
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,}")
        
        return True
        
    except Exception as e:
        print(f"模型测试失败: {e}")
        return False

if __name__ == "__main__":
    test_actual_model()