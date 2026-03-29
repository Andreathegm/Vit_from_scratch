import matplotlib.pyplot as plt
import torch
from data.transforms import IMAGENET_MEAN,IMAGENET_STD

def plot_single_rollout(img_np,mask,img_masked,save_path=None):
    _ , axes = plt.subplots(1, 3, figsize=(14, 5))
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
    if save_path is not None : 
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def plot_attention_grid(results: list, save_path = None,plot_mask=True):
    """
    Args:
        results: list of tuples (img_np, mask, img_masked)
                 one tuple per image —
    """
    if plot_mask:
        n_image_per_row = 3
    else:
        n_image_per_row = 2

    n = len(results)
    _ , axes = plt.subplots(n, n_image_per_row, figsize=(12, 4 * n))

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
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def denormalize(img: torch.Tensor,mean,std):
    """Inverts normalization for visualization."""
    img = img * std + mean              # denormalize
    img = img.clamp(0, 1)              
    return img.permute(1, 2, 0).numpy()  # (C,H,W) → (H,W,C) per matplotlib

def visualize_cutmix_mixup_augmentations(dataset_base, mixup_cutmix_ds, n_samples: int = 10,save_path = None):

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 3))
    fig.suptitle("Original  vs  MixUp/CutMix", fontsize=13, fontweight="bold")

    for i in range(n_samples):
        img_orig, label_orig = dataset_base[i]
        axes[i, 0].imshow(denormalize(img_orig,mean,std))
        axes[i, 0].set_title(f"Original — class {label_orig}")
        axes[i, 0].axis("off")

        img_aug, label_soft = mixup_cutmix_ds[i]

        dominant_class = label_soft.argmax().item()
        dominant_prob  = label_soft.max().item()
        axes[i, 1].imshow(denormalize(img_aug))
        axes[i, 1].set_title(
            f"Augmented — class {dominant_class} ({dominant_prob*100:.0f}%)"
        )
        axes[i, 1].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f"{save_path}.png", dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}.png")
    plt.show()




