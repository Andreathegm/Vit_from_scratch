import torch 
import os
def save_checkpoint(
    model, optimizer, scheduler,
    epoch, train_loss, val_loss, val_acc,
    best_val_acc, path
):
    torch.save({
        # stato del modello
        "model_state_dict": model.state_dict(),

        # stato dell'optimizer — momento primo e secondo ordine di AdamW
        # senza questo riparte da zero e le prime epoche sono instabili
        "optimizer_state_dict": optimizer.state_dict(),

        # stato dello scheduler — sa a che punto è il cosine decay
        # senza questo il lr riparte dal valore sbagliato
        "scheduler_state_dict": scheduler.state_dict(),

        # metriche e bookkeeping
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "best_val_acc": best_val_acc,
    }, path)
    print(f"Checkpoint salvato → {path} (epoch {epoch}, val_acc {val_acc*100:.2f}%)")

def load_checkpoint(path, model, optimizer, scheduler, device):
    """
    Carica il checkpoint e ripristina model, optimizer, scheduler.
    Restituisce l'epoca da cui riprendere e best_val_acc.
    """
    if not os.path.exists(path):
        print(f"Nessun checkpoint trovato in {path} — training da zero")
        return 0, 0.0, []

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    epoch         = checkpoint["epoch"]
    best_val_acc  = checkpoint["best_val_acc"]

    print(
        f"Checkpoint caricato da {path}\n"
        f"  Riparte da epoca {epoch + 1}\n"
        f"  Miglior val_acc finora: {best_val_acc*100:.2f}%"
    )

    # restituisce l'epoca successiva a quella salvata
    return epoch + 1, best_val_acc