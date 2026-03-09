import torch
from models.vit import VisionTransformer

def build_vit(config , device: torch.device):
    """
    Costruisce ViT-Tiny — versione ridotta di ViT-Base
    adatta per training from scratch su dataset medio-piccoli.

    Differenze rispetto a ViT-Base:
      embed_dim:  192  invece di 768   (4x meno parametri per layer)
      num_heads:  3    invece di 12    (head_dim = 192/3 = 64, invariato)
      depth:      12   invariato
    Totale: ~5.7M parametri invece di 86M
    """
    model = VisionTransformer(
        img_size = config.model.img_size,
        patch_size = config.model.patch_size,
        in_chans= config.model.in_chans,
        num_classes = config.dataset.num_classes,
        embed_dim = config.model.embed_dim, #192
        depth = config.model.depth, #12
        num_heads = config.model.num_heads, #3
        mlp_ratio = config.model.mlp_ratio,#4
        dropout = config.model.dropout,#0.1
        attn_dropout = config.model.attn_dropout,
        representation_size=None,  # nessun pre-logits layer — inutile from scratch
    ).to(device)

    # conta e stampa i parametri — utile per verificare che il modello
    # sia quello atteso prima di iniziare un training lungo
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")

    return model