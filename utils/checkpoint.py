import torch 
import os

CHECKPOINT_LAST = "checkpoints/last.pt"
CHECKPOINT_BEST = "checkpoints/best.pt"


def save_checkpoint(
    model, optimizer, scheduler,
    epoch, train_loss, val_loss, val_acc,
    best_val_acc, path
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch":                epoch,
        "train_loss":           train_loss,
        "val_loss":             val_loss,
        "val_acc":              val_acc,
        "best_val_acc":         best_val_acc,
    }, path)
    print(f"Checkpoint salvato → {path} (epoch {epoch}, val_acc {val_acc*100:.2f}%)")


def load_checkpoint(path, model, optimizer, scheduler, device, weights_only=False):
    """
    weights_only=False (default) → carica tutto: model, optimizer, scheduler
                                   usa per riprendere training normalmente
    weights_only=True            → carica solo i pesi del modello
                                   usa per fine-tuning con nuovo optimizer/scheduler
    """
    if not os.path.exists(path):
        print(f"Nessun checkpoint trovato in {path} — training da zero")
        return 0, 0.0

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if not weights_only:
        # scenario A — riprende tutto da dove si era fermato
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch        = checkpoint["epoch"]
    best_val_acc = checkpoint["best_val_acc"]

    print(
        f"Checkpoint caricato da {path} "
        f"{'(solo pesi)' if weights_only else '(completo)'}\n"
        f"  Epoca salvata:          {epoch}\n"
        f"  Miglior val_acc finora: {best_val_acc*100:.2f}%"
    )
    return epoch + 1, best_val_acc

