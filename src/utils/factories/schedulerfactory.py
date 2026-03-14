import torch

def build_Linear_and_CosineANL_scheduler(optimizer, epochs: int, warmup_epochs: int = 5,start_factor=0.01,end_factor=1.0):
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

def get_default_schedulers(optimezer,epochs,scheduler:str):
    match(scheduler.name):
        case "cosine":
            build_cosineannealingLR(optimezer,epochs,scheduler.eta_min)