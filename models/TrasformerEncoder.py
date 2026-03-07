import torch
import torch.nn as nn
from models.Attention import MultiHeadSelfAttention
from models.MLP import MLP

class TrasformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout=attn_dropout, proj_dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x: torch.Tensor):
        y, attn = self.attn(self.norm1(x))
        x = x + y
        y = self.mlp(self.norm2(x))
        x = x + y
        return x, attn

blk = TrasformerEncoderBlock(embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0)
x = torch.randn(64, 197, 768)
with torch.no_grad():
    y, a = blk(x)
print("EncoderBlock out:", y.shape)
print("EncoderBlock attn:", a.shape)