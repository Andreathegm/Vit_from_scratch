import torch
import torch.nn as nn
from models.patch_embedding import PatchEmbedding
from models.trasformerencoder import TrasformerEncoderBlock
from models.mlp import MLPHead


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        dropout: float = 0.1,
        representation_size = None,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size = img_size,in_channels=in_chans, patch_size=patch_size, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TrasformerEncoderBlock(embed_dim, num_heads, mlp_ratio, attn_dropout=attn_dropout, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = MLPHead(embed_dim, num_classes, representation_size=representation_size)

    def forward(self, x: torch.Tensor, return_attn: bool = False):
        x = self.patch_embed(x) # [B,197,768] = [batch, number of patches, embedding dim]
        x = self.pos_drop(x)

        attn_maps = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_maps.append(attn)

        x = self.norm(x)
        cls_token = x[:, 0] #[B, 768]
        logits = self.head(cls_token) # [B, number of classes]

        if return_attn:
            return logits, attn_maps
        return logits