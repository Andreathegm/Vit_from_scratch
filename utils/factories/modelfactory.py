import torch
from models.vit import VisionTransformer

def build_vit_tiny_model(img_size: int, patch_size: int, device: torch.device):
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
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=100,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0,
        representation_size=None,  # nessun pre-logits layer — inutile from scratch
    ).to(device)

    # conta e stampa i parametri — utile per verificare che il modello
    # sia quello atteso prima di iniziare un training lungo
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {total_params / 1e6:.2f}M")

    return model