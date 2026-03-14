import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size = 224,in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1,1,embed_dim))
        self.positional_embedding = nn.Parameter(torch.randn(1,self.num_patches+1,embed_dim))

    def forward(self, x):
        
        B, C, H, W = x.shape
        P = self.patch_size

        x = x.reshape(B, C, H // P, P, W // P, P)

        # ---> (B,num_patches_h,num_patches_w,C,P,P)
        x = x.permute(0, 2, 4, 1, 3, 5)
        
        # 3. FLATTEN
        x = x.reshape(B,self.num_patches, -1)
        
        # 4. LINEAR PROJECTION
        x = self.proj(x)

        cls = self.cls_token.expand(B,-1,-1)
        x = torch.cat([cls,x],dim=1)
        x = x + self.positional_embedding

        return x

# # --- ESEMPIO DI UTILIZZO ---
# # Immagine finta: Batch=2, Canali=3, Altezza=224, Larghezza=224
# dummy_images = torch.randn(2, 3, 224, 224)

# # Inizializziamo il modulo (i pesi vengono creati qui)
# patchifier = PatchEmbedding(patch_size=16, embed_dim=768)

# # Facciamo passare l'immagine (forward pass)
# patches = patchifier(dummy_images)