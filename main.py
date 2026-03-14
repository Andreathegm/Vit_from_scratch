import argparse
import torch
import torch.nn as nn

from src import load_yaml,get_device,build_vit_from_defaults,get_default_optimizers,get_default_schedulers,TrainSession,build_data_loaders_mixup,build_default_loaders,load_weights_from_complex_checkpoint,get_default_criterions,build_dataloaders,get_default_evaluation_action


CHOICES = ["train", "test"]

def parse_args():
    parser = argparse.ArgumentParser(description="Vision Transformer Training & Evaluation")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help=".yaml file to use for training"
    )

    parser.add_argument(
        "--mode", 
        type=str,
        required=True,
        choices=CHOICES, 
        help="argument necessary to know wheter the model needs to be evaluated or trained"
    )
    
    return parser.parse_args()


def main():
    # 1. Setup Iniziale
    args = parse_args()
    config = load_yaml(args.config)
    device = get_device()
    
    print(f"Starting in {args.mode.upper()} mode")
    print(f"Configuration taken from {args.config}")

    model = build_vit_from_defaults(config.model)
    print(model)
    criterion = get_default_criterions(config.criterion)
    print(config.criterion.label_smoothing)
    print(criterion)


    # ==========================================
    # TRAINING MODE
    # ==========================================
    match(args.mode):
        case "train":
            # optimizer = torch.optim.AdamW(model.parameters(), weight_decay=config.training.weight_decay)
            optimizer = get_default_optimizers(config.optimizer)
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs, eta_min=1e-6)
            scheduler = get_default_schedulers(config.scheduler)
            # criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

            session = TrainSession(
                model, optimizer, scheduler, criterion,
                config = config,
                run_name = config.run_name,
                epochs = config.epochs,
                device = device,
                weights_path = config.weights_path
            )
            
            # Carica stato ottimizzatore se necessario (aggiungi la logica nello YAML)
            if config.load_optim_state:
                session.load_optimizer_state(config.weights_path, new_lr=config.lr)
                
            print(session)

            # Costruisce i dataloader giusti in base a un parametro nello YAML
            if getattr(config.training, "use_mixup", False):
                print("Building mixup-cutmix dataloader")
                train_loader, val_loader, test_loader = build_data_loaders_mixup(config.model.img_size, config.training.batch_size)
            else:
                print("Using Standard dataloaders")
                train_loader, val_loader, test_loader = build_dataloaders(config.model.img_size, config.training.batch_size)

            session.train_and_test(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

        # ==========================================
        # EVALUATION MODE
        # ==========================================
        case "test":
            weights_path = config.weights_path
            model = load_weights_from_complex_checkpoint(model, weights_path, device, strict=True)
            loader = build_default_loaders(config.img_size,config.batch_size, config.split)
            evaluation_action = get_default_evaluation_action(config.k)
            evaluation_action(model=model, loader=loader, criterion=criterion, device=device, split=config.split,k=config.k)       

if __name__ == "__main__":
    main()