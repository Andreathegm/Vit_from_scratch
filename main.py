import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time

from utils.device import get_device
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    CHECKPOINT_LAST,
    CHECKPOINT_BEST
)
from utils.wandb_logger import (
    init_wandb,
    log_epoch,
    log_best,
    log_test,
    finish_wandb
)
from train import train_one_epoch, evaluate
from config.config import Hyperparams as Hp
from dataset import ImageDataset, TransformDataset, get_transforms
from models.vit import VisionTransformer


def build_dataloaders(img_size: int, batch_size: int):
    """
    Costruisce train, val e test loader.
    Separato dal main per tenere il codice leggibile —
    la logica dei dataset non appartiene al training loop.
    """

    # carica il dataset grezzo senza transform —
    # la transform viene applicata dopo separatamente su train e val
    # per evitare il bug di condivisione dello stesso ImageFolder
    raw_ds = ImageDataset("data/imagenet-100/train")

    # split stratificato — ogni classe ha la stessa proporzione
    # in train e val, evita che classi rare finiscano solo in uno dei due
    train_subset, val_subset = raw_ds.split(val_ratio=0.1)

    # TransformDataset applica la transform giusta a ciascun subset
    # in modo completamente indipendente
    train_ds = TransformDataset(
        train_subset,
        transform=get_transforms(img_size, "train"),
        classes=raw_ds.classes
    )
    val_ds = TransformDataset(
        val_subset,
        transform=get_transforms(img_size, "val"),
        classes=raw_ds.classes
    )

    # test set — dataset separato, nessuno split necessario
    # la transform viene passata direttamente nel costruttore
    test_ds = ImageDataset(
        "data/imagenet-100/val.X",
        transform=get_transforms(img_size, "test")
    )

    # pin_memory=True — i tensori vengono allocati in memoria pinned
    # che permette trasferimento CPU→GPU asincrono con non_blocking=True
    # shuffle=False su val e test — l'ordine non influenza la valutazione
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,       # 4 worker per non fare aspettare la GPU
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def build_model(img_size: int, patch_size: int, device: torch.device):
    """
    Costruisce ViT-Tiny — versione ridotta di ViT-Base
    adatta per training from scratch su dataset medio-piccoli.

    Differenze rispetto a ViT-Base:
      embed_dim:  192  invece di 768   (4x meno parametri per layer)
      num_heads:  3    invece di 12    (head_dim = 192/3 = 64, invariato)
      depth:      12   invariato
    Totale: ~5.7M parametri invece di 86M
    """
    model = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=3,
        num_classes=100,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        dropout=0.1,
        attn_dropout=0.0,
        representation_size=None,  # nessun pre-logits layer — inutile from scratch
    ).to(device)

    # conta e stampa i parametri — utile per verificare che il modello
    # sia quello atteso prima di iniziare un training lungo
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri totali: {total_params / 1e6:.2f}M")

    return model


def build_scheduler(optimizer, epochs: int, warmup_epochs: int = 5):
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


def main():
    os.makedirs("checkpoints", exist_ok=True)

    device     = get_device()
    img_size   = Hp.IMG_SIZE
    patch_size = Hp.PATCH_SIZE
    batch_size = Hp.BATCH_SIZE
    EPOCHS     = 100
    WARMUP     = 5

    # costruisce i loader in una funzione separata —
    # il main si occupa solo del training loop
    train_loader, val_loader, test_loader = build_dataloaders(img_size, batch_size)

    model = build_model(img_size, patch_size, device)

    # AdamW — Adam con weight decay corretto
    # il weight decay in Adam standard è applicato in modo sbagliato
    # AdamW lo applica direttamente ai pesi, non al gradiente
    # lr=1e-3 e weight_decay=0.05 — valori dal DeiT paper
    # Touvron et al. 2021, Tabella 9
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.05
    )

    # CrossEntropyLoss con label_smoothing=0.1 —
    # invece di target one-hot [0,0,1,0] usa [0.033,0.033,0.9,0.033]
    # previene che il modello diventi troppo sicuro e migliora generalizzazione
    # usato nel ViT paper — Dosovitskiy et al. 2020, Appendice B
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scheduler = build_scheduler(optimizer, EPOCHS, WARMUP)

    # config salvata su wandb — permette di confrontare run diversi
    # e sapere esattamente con quali iperparametri è stato ottenuto ogni risultato
    config = {
        "model":        "ViT-Tiny",
        "epochs":       EPOCHS,
        "warmup_epochs": WARMUP,
        "batch_size":   batch_size,
        "lr":           1e-3,
        "weight_decay": 0.05,
        "embed_dim":    192,
        "depth":        12,
        "num_heads":    3,
        "mlp_ratio":    4.0,
        "dropout":      0.1,
        "img_size":     img_size,
        "patch_size":   patch_size,
        "label_smoothing": 0.1,
        "augmentation": "upresize&randomcrop+ColorJitter",
        "scheduler":    f"warmup{WARMUP}+cosine",
        "dataset":      "ImageNet-100",
        "num_classes":  100,
    }
    init_wandb(config, run_name="vit-tiny-200ep")

    # prova a riprendere dall'ultimo checkpoint —
    # se non esiste start_epoch=0 e best_val_acc=0.0
    # se esiste carica model, optimizer e scheduler
    # e restituisce l'epoca successiva all'ultima salvata
    start_epoch, best_val_acc = load_checkpoint(
        CHECKPOINT_LAST, model, optimizer, scheduler, device
    )

    start      = time()
    epoch_pbar = tqdm(range(start_epoch, EPOCHS + 1), desc="Training")

    for epoch in epoch_pbar:

        # train_one_epoch ritorna la loss media sull'intero epoch
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch, device
        )

        # evaluate su val — NON su test
        # il test set viene toccato una sola volta alla fine
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, split="Val"
        )

        # scheduler.step() va chiamato dopo evaluate —
        # aggiorna il lr per l'epoca successiva
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # loga su wandb — aggiorna la dashboard in tempo reale
        log_epoch(epoch, train_loss, val_loss, val_acc, current_lr)

        # salva sempre l'ultimo checkpoint dopo ogni epoca —
        # se il training si interrompe si riparte da qui
        # sovrascrive il checkpoint precedente — tiene solo l'ultimo
        save_checkpoint(
            model, optimizer, scheduler,
            epoch, train_loss, val_loss, val_acc,
            best_val_acc, CHECKPOINT_LAST
        )

        # salva il miglior checkpoint separatamente —
        # non viene mai sovrascritto da checkpoint peggiori
        # alla fine del training carichiamo questo per il test
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, train_loss, val_loss, val_acc,
                best_val_acc, CHECKPOINT_BEST
            )
            log_best(best_val_acc, epoch)

        epoch_pbar.set_postfix(
            train=f"{train_loss:.4f}",
            val=f"{val_loss:.4f}",
            acc=f"{val_acc*100:.2f}%",
            best=f"{best_val_acc*100:.2f}%",
            lr=f"{current_lr:.2e}"
        )

    elapsed = (time() - start) / 60
    print(f"\nTraining completato in {elapsed:.1f} minuti")

    # carica il miglior modello — NON l'ultimo
    # l'ultimo potrebbe aver overfittato nelle ultime epoche
    print("Carico il miglior modello per il test finale...")
    load_checkpoint(CHECKPOINT_BEST, model, optimizer, scheduler, device)

    # test set — tocca UNA SOLA VOLTA qui
    # usarlo durante il training invalida la valutazione
    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, split="Test"
    )

    print(f"Test Loss:    {test_loss:.4f}")
    print(f"Test Acc:     {test_acc*100:.2f}%")
    print(f"Best Val Acc: {best_val_acc*100:.2f}%")

    log_test(test_acc, test_loss)
    finish_wandb()


if __name__ == "__main__":
    main()