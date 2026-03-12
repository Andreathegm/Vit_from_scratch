from trainsession import TrainSession
from utils.factories.modelfactory import build_vit
from utils.factories.dataloaderfactory import build_dataloaders,build_train_val_loader,build_data_loaders_mixup
from config.config import load_yaml
from utils.device import get_device
import torch
import torch.nn as nn

def run_run1():
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

def run_evaluate_train_accuracy():
    path = "config/run_eval_train.yaml"
    weights = "checkpoints/vit-fine-tune-mixup-run3/best.pt"
    config = load_yaml(path)
    device = get_device()
    model = build_vit(config=config,device=device)
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    from train import  evaluate
    train_val_loader = build_train_val_loader(config.model.img_size,config.training.batch_size)
    evaluate(model=model,loader= train_val_loader,criterion = criterion, device=device,split="Train Accuracy")
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
def run_evaluate_test_top_k(k):
    path = "config/run_eval_train.yaml"
    weights = "checkpoints/vit-fine-tune-mixup/best.pt"
    config = load_yaml(path)
    device = get_device()
    model = build_vit(config=config,device=device)
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    from train import  evaluate_top_k
    _,_,test_dl = build_dataloaders(config.model.img_size,config.training.batch_size)
    evaluate_top_k(model=model,loader=test_dl,criterion = criterion, device=device,split="Train Accuracy",k=k)

def run_run2():
    path = "config/run2.yaml"
    config = load_yaml(path)
    device = get_device()
    model = build_vit(config=config,device=device)


    optimizer = torch.optim.AdamW(model.parameters(),weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs,eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
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
    train_loader,val_loader,test_loader = build_data_loaders_mixup(config.model.img_size,config.training.batch_size)
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

def run_run3():
    path = "config/run3finetuneofmixup.yaml"
    config = load_yaml(path)
    device = get_device()
    model = build_vit(config=config,device=device,echo=True)


    optimizer = torch.optim.AdamW(model.parameters(),weight_decay=config.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.training.epochs,eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    session = TrainSession(
        model, optimizer, scheduler, criterion,
        config=config,
        run_name=config.training.run_name,
        epochs=config.training.epochs,
        device=device,
        weights_path=config.training.weights_path
        )
    print(session)
    session.load_optimizer_state(config.training.weights_path, new_lr = config.training.lr,no_wd_load=True)
    # for group in session.optimizer.param_groups:
    #     print (group.keys())
    # return
    print(session)

    train_loader,val_loader,test_loader = build_data_loaders_mixup(config.model.img_size,config.training.batch_size)
    # print(len(train_loader.dataset),len(val_loader.dataset),len(test_loader.dataset))
    # return
    session.train_and_test(train_loader=train_loader, val_loader=val_loader,test_loader=test_loader)