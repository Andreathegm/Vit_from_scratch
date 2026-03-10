from trainsession import TrainSession
from runs import run_evaluate_train_accuracy,run_run2
from utils.factories.modelfactory import build_vit
from utils.factories.dataloaderfactory import build_dataloaders
from config.config import load_yaml
from utils.device import get_device
import torch
import torch.nn as nn



# def main():

#     os.makedirs("checkpoints", exist_ok=True)

#     device     = get_device()
#     img_size   = Hp.IMG_SIZE
#     patch_size = Hp.PATCH_SIZE
#     batch_size = Hp.BATCH_SIZE
#     EPOCHS     = 100
#     WARMUP     = 5

#     # costruisce i loader in una funzione separata —
#     # il main si occupa solo del training loop
#     train_loader, val_loader, test_loader = build_dataloaders(img_size, batch_size)

#     model = build_model(img_size, patch_size, device)
#     # CrossEntropyLoss con label_smoothing=0.1 —
#     # invece di target one-hot [0,0,1,0] usa [0.033,0.033,0.9,0.033]
#     # previene che il modello diventi troppo sicuro e migliora generalizzazione
#     # usato nel ViT paper — Dosovitskiy et al. 2020, Appendice B
#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

#     # AdamW — Adam con weight decay corretto
#     # AdamW lo applica direttamente ai pesi, non al gradiente
#     # lr=1e-3 e weight_decay=0.05 — valori dal DeiT paper
#     # Touvron et al. 2021, Tabella 9
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=1e-3,
#         weight_decay=0.05
#     )


#     scheduler = build_scheduler(optimizer, EPOCHS, WARMUP)

#     # config salvata su wandb — permette di confrontare run diversi
#     # e sapere esattamente con quali iperparametri è stato ottenuto ogni risultato
#     config = {
#         "model":        "ViT-Tiny",
#         "epochs":       EPOCHS,
#         "warmup_epochs": WARMUP,
#         "batch_size":   batch_size,
#         "lr":           1e-3,
#         "weight_decay": 0.05,
#         "embed_dim":    192,
#         "depth":        12,
#         "num_heads":    3,
#         "mlp_ratio":    4.0,
#         "dropout":      0.1,
#         "img_size":     img_size,
#         "patch_size":   patch_size,
#         "label_smoothing": 0.1,
#         "augmentation": "upresize&randomcrop+ColorJitter",
#         "scheduler":    f"warmup{WARMUP}+cosine",
#         "dataset":      "ImageNet-100",
#         "num_classes":  100,
#     }
#     init_wandb(config, run_name="vit-tiny-200ep")

#     # prova a riprendere dall'ultimo checkpoint —
#     # se non esiste start_epoch=0 e best_val_acc=0.0
#     # se esiste carica model, optimizer e scheduler
#     # e restituisce l'epoca successiva all'ultima salvata
#     start_epoch, best_val_acc = load_checkpoint(
#         CHECKPOINT_LAST, model, optimizer, scheduler, device
#     )

#     start      = time()
#     epoch_pbar = tqdm(range(start_epoch, EPOCHS + 1), desc="Training")

#     for epoch in epoch_pbar:

#         # train_one_epoch ritorna la loss media sull'intero epoch
#         train_loss = train_one_epoch(
#             model, train_loader, optimizer, criterion, epoch, device
#         )

#         # evaluate su val — NON su test
#         # il test set viene toccato una sola volta alla fine
#         val_loss, val_acc = evaluate(
#             model, val_loader, criterion, device, split="Val"
#         )

#         # scheduler.step() va chiamato dopo evaluate —
#         # aggiorna il lr per l'epoca successiva
#         scheduler.step()

#         current_lr = optimizer.param_groups[0]["lr"]

#         # loga su wandb — aggiorna la dashboard in tempo reale
#         log_epoch(epoch, train_loss, val_loss, val_acc, current_lr)

#         # salva sempre l'ultimo checkpoint dopo ogni epoca —
#         # se il training si interrompe si riparte da qui
#         # sovrascrive il checkpoint precedente — tiene solo l'ultimo
#         save_checkpoint(
#             model, optimizer, scheduler,
#             epoch, train_loss, val_loss, val_acc,
#             best_val_acc, CHECKPOINT_LAST
#         )

#         # salva il miglior checkpoint separatamente —
#         # non viene mai sovrascritto da checkpoint peggiori
#         # alla fine del training carichiamo questo per il test
#         if val_acc > best_val_acc:
#             best_val_acc = val_acc
#             save_checkpoint(
#                 model, optimizer, scheduler,
#                 epoch, train_loss, val_loss, val_acc,
#                 best_val_acc, CHECKPOINT_BEST
#             )
#             log_best(best_val_acc, epoch)

#         epoch_pbar.set_postfix(
#             train=f"{train_loss:.4f}",
#             val=f"{val_loss:.4f}",
#             acc=f"{val_acc*100:.2f}%",
#             best=f"{best_val_acc*100:.2f}%",
#             lr=f"{current_lr:.2e}"
#         )

#     elapsed = (time() - start) / 60
#     print(f"\nTraining completato in {elapsed:.1f} minuti")

#     # carica il miglior modello — NON l'ultimo
#     # l'ultimo potrebbe aver overfittato nelle ultime epoche
#     print("Carico il miglior modello per il test finale...")
#     load_checkpoint(CHECKPOINT_BEST, model, optimizer, scheduler, device)

#     # test set — tocca UNA SOLA VOLTA qui
#     # usarlo durante il training invalida la valutazione
#     test_loss, test_acc = evaluate(
#         model, test_loader, criterion, device, split="Test"
#     )

#     print(f"Test Loss:    {test_loss:.4f}")
#     print(f"Test Acc:     {test_acc*100:.2f}%")
#     print(f"Best Val Acc: {best_val_acc*100:.2f}%")

#     log_test(test_acc, test_loss)
#     finish_wandb()
def main():
    run_run2()
    return
    run_evaluate_train_accuracy()
    return
    path = "config/run1.yaml"
    config = load_yaml(path)
    device = get_device()
    model = build_vit(config=config,device=device)


    optimizer = torch.optim.AdamW(model.parameters(),weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs // 2,eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    session = TrainSession(
        model, optimizer, scheduler, criterion,
        config=config,
        run_name=config.training.run_name,
        epochs=config.training.epochs,
        device=device,
        weights_path=config.training.weights_path
        )
    print(session)
    session.load_optimizer_state(config.training.weights_path, new_lr = config.training.lr)
    print(session)
    train_loader,val_loader,test_loader = build_dataloaders(config.model.img_size,config.training.batch_size)
    # print(train_loader,val_loader,test_loader)
    # print(next(session.model.parameters()).device)
    # x, y = next(iter(train_loader))
    # print(x.shape)   # (batch_size, 3, 224, 224)
    # print(y.shape)   # (batch_size,)
        # train deve avere RandAugment, RandomErasing ecc
    # val/test deve essere solo Resize + CenterCrop + Normalize
    # print(train_loader.dataset.transform)
    # print(val_loader.dataset.transform)
    # print(vars(test_loader.dataset))
    # return
    session.train_and_test(train_loader=train_loader, val_loader=val_loader,test_loader=test_loader)

if __name__ == "__main__":
    main()