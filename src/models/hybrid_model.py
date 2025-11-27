import torch
import torch.nn as nn

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm hỗ trợ định dạng (N, C, H, W)"""
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x

class ConvNeXtBlock(nn.Module):
    """Khối cơ bản của ConvNeXt"""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class BrainTumorModel(nn.Module):
    """
    Mô hình lai ghép: ConvNeXt-Tiny Backbone + Transformer Encoder
    """
    def __init__(self, num_classes=4):
        super().__init__()
        dims = [96, 192, 384, 768] 
        
        # CNN Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0])
        )
        
        self.stage1 = nn.Sequential(*[ConvNeXtBlock(dims[0]) for _ in range(3)])
        
        self.downsample1 = nn.Sequential(LayerNorm2d(dims[0]), nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2))
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(dims[1]) for _ in range(3)])
        
        self.downsample2 = nn.Sequential(LayerNorm2d(dims[1]), nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2))
        self.stage3 = nn.Sequential(*[ConvNeXtBlock(dims[2]) for _ in range(3)])
        
        self.downsample3 = nn.Sequential(LayerNorm2d(dims[2]), nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2))
        self.stage4 = nn.Sequential(*[ConvNeXtBlock(dims[3]) for _ in range(3)])
        
        # Transformer Encoder
        self.norm_final = nn.LayerNorm(dims[3])
        encoder_layer = nn.TransformerEncoderLayer(d_model=dims[3], nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classifier Head
        self.head = nn.Linear(dims[3], num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.downsample1(x)
        x = self.stage2(x)
        x = self.downsample2(x)
        x = self.stage3(x)
        x = self.downsample3(x)
        x = self.stage4(x)
        
        # Flatten cho Transformer
        x = x.permute(0, 2, 3, 1) 
        x = x.flatten(1, 2) 
        x = self.norm_final(x)
        
        x = self.transformer(x)
        x = x.mean(dim=1) # Global Average Pooling
        x = self.head(x)
        return x