import wandb

def init_wandb(config: dict, run_name: str, project: str = "vit-imagenet100"):
    """
    Inizializza un run wandb.
    resume='allow' riprende il run esistente se ha lo stesso nome —
    fondamentale per quando riprendi il training da un checkpoint.
    """
    wandb.init(
        project=project,
        name=run_name,
        resume="allow",       # se esiste un run con questo nome lo riprende
        config=config,        # salva tutti gli iperparametri
    )
    print(f"wandb run inizializzato → {wandb.run.url}")


def log_epoch(epoch, train_loss, val_loss, val_acc, lr):
    """
    Loga le metriche di una epoca.
    step=epoch rende l'asse x della dashboard leggibile.
    """
    wandb.log({
        "epoch":      epoch,
        "train_loss": train_loss,
        "val_loss":   val_loss,
        "val_acc":    val_acc * 100,   # in percentuale — più leggibile
        "lr":         lr,
    }, step=epoch)


def log_best(best_val_acc, best_epoch):
    """
    Aggiorna il summary — appare nella tabella dei run
    come il valore più importante di quel run.
    """
    wandb.run.summary["best_val_acc"] = best_val_acc * 100
    wandb.run.summary["best_epoch"]   = best_epoch


def log_test(test_acc, test_loss):
    wandb.run.summary["test_acc"]  = test_acc * 100
    wandb.run.summary["test_loss"] = test_loss


def finish_wandb():
    wandb.finish()