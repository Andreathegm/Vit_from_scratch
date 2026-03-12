from src import load_yaml
from utils.device import get_device
from models import VisionTransformer

def build_vit(config ,echo=False):

    model = VisionTransformer(
        img_size = config.img_size,
        patch_size = config.patch_size,
        in_chans= config.in_chans,
        num_classes = config.num_classes,
        embed_dim = config.embed_dim, #192
        depth = config.depth, #12
        num_heads = config.num_heads, #3
        mlp_ratio = config.mlp_ratio,#4
        dropout = config.dropout,#0.1
        attn_dropout = config.attn_dropout,
        representation_size=None,
    ).to(get_device())

    if echo:
      print(model)
      total_params = sum(p.numel() for p in model.parameters())
      print(f"Total parameters: {total_params / 1e6:.2f}M")

    return model

def build_vit_tiny224_16(echo=False):
  """
      Builds Vit-tiny model

      Differences respect to ViT-Base:

        embed_dim:  192       768   (4x less param per layer)
        num_heads:  3         12    (head_dim = 192/3 = 64, not changed )
        depth:      12        12
        
      Total:      ~5.54M       86M
  """
  yaml_path = "configs/model/Vit-Tiny.yaml"
  config = load_yaml(yaml_path)
  build_vit(config,echo)

def build_vit_base224_16(echo=False):
   pass