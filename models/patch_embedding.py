import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        
        # La dimensione del vettore appiattito sarà: Canali * P * P
        # Per un'immagine RGB (3 canali) con patch 16x16, sarà 3 * 16 * 16 = 768
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Il layer lineare che sostituisce Conv2d
        self.proj = nn.Linear(self.patch_dim, embed_dim)

    def forward(self, x):
        print("1. Input tensor shape (B, C, H, W):", x.shape)
        
        B, C, H, W = x.shape
        P = self.patch_size
        
        # Assicuriamoci che l'immagine sia divisibile in patch esatte
        assert H % P == 0 and W % P == 0, "Dimensioni immagine non divisibili per patch_size"
                
        num_patches_h = H // P
        num_patches_w = W // P
        
        # 1 & 2. CREAZIONE DELLE PATCH
        # Rimodelliamo per isolare i blocchi PxP
        x = x.view(B, C, num_patches_h, P, num_patches_w, P)
        
        # Scambiamo le dimensioni per avere: (Batch, Griglia_H, Griglia_W, Canali, Patch_H, Patch_W)
        x = x.permute(0, 2, 4, 1, 3, 5)
        print("2. Dopo l'estrazione delle patch (B, Grid_H, Grid_W, C, P, P):", x.shape)
        
        # 3. FLATTEN
        # Combiniamo le griglie H e W nella sequenza N (numero totale di patch)
        # Combiniamo i canali e i pixel PxP nel singolo vettore patch_dim
        x = x.contiguous().view(B, num_patches_h * num_patches_w, self.patch_dim)
        print("3. Dopo il flatten (B, N, C*P*P):", x.shape)
        
        # 4. LINEAR PROJECTION
        x = self.proj(x)
        print("4. Dopo Linear Projection (B, N, D):", x.shape)
        
        return x

# --- ESEMPIO DI UTILIZZO ---
# Immagine finta: Batch=2, Canali=3, Altezza=224, Larghezza=224
dummy_images = torch.randn(2, 3, 224, 224)

# Inizializziamo il modulo (i pesi vengono creati qui)
patchifier = PatchEmbedding(patch_size=16, embed_dim=768)

# Facciamo passare l'immagine (forward pass)
patches = patchifier(dummy_images)