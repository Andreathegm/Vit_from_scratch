import torch
import numpy as np 
from PIL import Image

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

def visualize_attention_rollout(model, image_path: str, device, patch_size: int = 16, img_size: int = 224):
    
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



    return img_np,mask,img_masked