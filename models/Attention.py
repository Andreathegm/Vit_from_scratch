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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)  # one projection for Q, K, V
        self.attn = ScaledDotProductAttention(dropout=attn_dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)     # output projection
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor):
        B, N, D = x.shape

        qkv = self.qkv(x)              # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1) # each is [B, N, D]

        q = q.view(B, N, self.num_heads, self.head_dim)
        k = k.view(B, N, self.num_heads, self.head_dim)
        v = v.view(B, N, self.num_heads, self.head_dim)

        # transpose moves the head dimension before sequence length so attention is computed per head
        q = q.transpose(1, 2)  # [B, h, N, d]
        k = k.transpose(1, 2)  # [B, h, N, d]
        v = v.transpose(1, 2)  # [B, h, N, d]

        out, attn = self.attn(q, k, v)

        # merge heads back to D
        out = out.transpose(1, 2).contiguous()  # contiguous makes memory layout compatible with view
        out = out.view(B, N, D)

        out = self.proj(out)
        out = self.proj_dropout(out)
        return out, attn

mhsa = MultiHeadSelfAttention(embed_dim=768, num_heads=12)
x = torch.randn(64, 197, 768)
with torch.inference_mode():
    y, a = mhsa(x)
print("MultiHeadSelfAttention out:", y.shape)
print("MultiHeadSelfAttention attn:", a.shape)