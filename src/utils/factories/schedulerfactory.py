import torch

def build_Linear_and_CosineANL_scheduler(optimizer, epochs: int, warmup_epochs: int = 5,start_factor=0.2,end_factor=1.0):
    """
    linear Warmup + CosineAnnealing
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=start_factor,   # lr iniziale = lr_target * 0.01
        end_factor=end_factor,      # lr finale del warmup = lr_target
        total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs - warmup_epochs  # epoche rimanenti dopo il warmup
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs]
    )

def build_cosineannealingLR(optimizer,epochs,eta_min):

    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=eta_min  
    )

def build_cosineannealing_with_middle_restart(optimizer,epochs):
    return  torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=epochs // 2 ,      # durata del ciclo
    T_mult=1,    # non allungare i cicli
    eta_min=0.0  # minimo del coseno
)

def build_reduce_on_plateau_scheduler(optimizer, patience, factor: float = 0.5, min_lr: float = 1e-6):
    """ReduceLROnPlateau — decreases lr when val_acc stops improving."""
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=factor,
        patience=patience,
        min_lr=min_lr
    )
def get_default_schedulers(optimezer,scheduler:str):
    match(scheduler.name):

        case "cosine":
            return build_cosineannealingLR(optimezer,scheduler.epochs,scheduler.eta_min)
        
        case "ReduceLROnPlateau":

            return build_reduce_on_plateau_scheduler(optimizer=optimezer,patience=scheduler.patience,min_lr=scheduler.min_lr,factor=scheduler.factor)
        
        case "cosine_mr":
            return build_cosineannealing_with_middle_restart(optimizer=optimezer,epochs=scheduler.epochs)

        case "linear+cosine":
            return build_Linear_and_CosineANL_scheduler(optimizer=optimezer,epochs=scheduler.epochs)