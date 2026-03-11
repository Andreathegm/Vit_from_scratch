import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils.factories.modelfactory import build_vit
from utils.device import get_device
from utils.folders import random_element_from_subfolders
from config.config import load_yaml


def attention_rollout(attn_maps: list, patch_size: int = 16, img_size: int = 224):
    
    num_patches_side = img_size // patch_size
    
    rollout = None
    for attn in attn_maps:
        attn_mean = attn[0].mean(dim=0)
        attn_mean = attn_mean + torch.eye(attn_mean.shape[0], device=attn_mean.device)
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
        
        if rollout is None:
            rollout = attn_mean
        else:
            rollout = torch.matmul(attn_mean, rollout)
    
    mask = rollout[0, 1:]
    mask = mask.reshape(num_patches_side, num_patches_side).cpu().numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = np.array(Image.fromarray(mask).resize((img_size, img_size), resample=Image.BILINEAR))
    
    return mask

def visualize_attention_rollout(model, image_path: str, device, patch_size: int = 16, img_size: int = 224,visualize=False):
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    transform = T.Compose([
        T.Resize(int(img_size * (240 / 224))),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

    img_pil    = Image.open(image_path).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    img_np = (img_tensor.squeeze(0).cpu() * std + mean)
    img_np = img_np.clamp(0, 1).permute(1, 2, 0).numpy()

    model.eval()
    with torch.no_grad():
        _, attn_maps = model(img_tensor, return_attn=True)

    mask       = attention_rollout(attn_maps, patch_size, img_size)
    img_masked = img_np * mask[:, :, np.newaxis]


    if not visualize:

        return img_np,mask,img_masked
    
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Attention Mask")
        axes[1].axis("off")

        axes[2].imshow(img_masked)
        axes[2].set_title("Attention Map")
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig("plots/attention_rollout.png", dpi=150, bbox_inches="tight")
        plt.show()

def plot_attention_grid(results: list, save_path: str = "plots/attention_grid.png"):
    """
    Args:
        results: list of tuples (img_np, mask, img_masked)
                 one tuple per image — returned by get_attention_rollout
    """
    n    = len(results)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))

    for i, (img_np, mask, img_masked) in enumerate(results):
        axes[i, 0].imshow(img_np)
        axes[i, 0].set_title("Original")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Attention Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(img_masked)
        axes[i, 2].set_title("Attention Map")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

yaml_path = "config/run2.yaml"
weights = "checkpoints/vit-fine-tune-mixup/best.pt"
config = load_yaml(yaml_path)
device = get_device()
model = build_vit(config=config,device=device)
checkpoint = torch.load(weights, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

seed = 1 
img_path_root = "data/imagenet-100/train"
imgs_results = []
for i in range (8):
    img_path = random_element_from_subfolders(img_path_root,seed+i)
    img_np,mask,img_masked = visualize_attention_rollout(model,img_path,device)
    imgs_results.append((img_np,mask,img_masked))
plot_attention_grid(imgs_results)


