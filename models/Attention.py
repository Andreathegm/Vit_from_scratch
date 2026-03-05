import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q , k , v): 
        d_k = q.shape[-1]

        # k.transpose(-2, -1) turns [B, h, N, d] into [B, h, d, N] so we can do Q @ K^T
        scores = q @ k.transpose(-2, -1)  # [B, h, N, N]
        scores = scores / math.sqrt(d_k)  # scaling keeps scores in a good numeric range

        attn = F.softmax(scores, dim=-1)  # attention weights over keys
        attn = self.dropout(attn)

        out = attn @ v  # weighted sum of values
        return out, attn

attn_layer = ScaledDotProductAttention(dropout=0.0)
q = torch.randn(64, 12, 197, 64)
k = torch.randn(64, 12, 197, 64)
v = torch.randn(64, 12, 197, 64)
with torch.inference_mode():
    out, attn = attn_layer(q, k, v)
print("ScaledDotProductAttention out:", out.shape)
print("ScaledDotProductAttention attn:", attn.shape)