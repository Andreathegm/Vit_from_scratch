import torch

def build_Linear_and_CosineANL_scheduler(optimizer, epochs: int, warmup_epochs: int = 5):
    """
    Warmup lineare + CosineAnnealing — schema standard per ViT.

    Warmup — nelle prime epoche il learning rate parte basso (lr * 0.01)
    e cresce linearmente fino al valore target. Senza warmup i gradienti
    nelle prime epoche sono instabili e il modello diverge.
    Fonte: Dosovitskiy et al. 2020, Appendice B.

    CosineAnnealing — dopo il warmup il lr decade seguendo una curva coseno
    fino a zero. Più smooth di StepLR, evita drop bruschi del lr.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.01,   # lr iniziale = lr_target * 0.01
        end_factor=1.0,      # lr finale del warmup = lr_target
        total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs  # epoche rimanenti dopo il warmup
    )

    # SequentialLR applica warmup per le prime warmup_epochs,
    # poi passa automaticamente a cosine
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )