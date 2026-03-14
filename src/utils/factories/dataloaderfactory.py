from torch.utils.data import DataLoader
from data.dataset import MixUpCutMixDataset,ImageDataset,TransformDataset
from data.transforms import get_transforms

def build_dataloaders(img_size: int, batch_size: int,train_val_ratio=0.1):

    raw_ds = ImageDataset("dataset/imagenet-100/train")

    train_subset, val_subset = raw_ds.split(val_ratio=train_val_ratio)

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

    test_ds = ImageDataset(
        "dataset/imagenet-100/val.X",
        transform=get_transforms(img_size, "test")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,       
        pin_memory=True,
        persistent_workers=True,  
        prefetch_factor=2       
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,  # evita di ricreare i worker ad ogni epoca
        prefetch_factor=2  
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,  # evita di ricreare i worker ad ogni epoca
        prefetch_factor=2  
    )

    return train_loader, val_loader, test_loader

def build_train_eval_loader(img_size: int, batch_size: int):
   
    raw_ds = ImageDataset("dataset/imagenet-100/train")

    train_subset, _ = raw_ds.split(val_ratio=0.1)

    train_ds = TransformDataset(
        train_subset,
        transform=get_transforms(img_size, "train.eval"),
        classes=raw_ds.classes
    )

    train_eval_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,      
        persistent_workers=True,  
        prefetch_factor=2         
    )
  

    return train_eval_loader

def build_data_loaders_mixup(img_size: int, batch_size: int):

    raw_ds = ImageDataset("dataset/imagenet-100/train")

    train_subset, val_subset = raw_ds.split(val_ratio=0.1)

    train_ds_base = TransformDataset(
        train_subset,
        transform=get_transforms(img_size, "train.eval"),
        classes=raw_ds.classes
    )

    # wrappa con MixUp/CutMix
    train_ds = MixUpCutMixDataset(
        train_ds_base,
        num_classes=100,
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        prob=0.5,
        switch_prob=0.5
    )

    val_ds = TransformDataset(
        val_subset,
        transform=get_transforms(img_size, "val"),
        classes=raw_ds.classes
    )

    # test set — dataset separato, nessuno split necessario
    # la transform viene passata direttamente nel costruttore
    test_ds = ImageDataset(
        "dataset/imagenet-100/val.X",
        transform=get_transforms(img_size, "test")
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,      
        persistent_workers=True,
        prefetch_factor=2        
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True,  
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        persistent_workers=True, 
        prefetch_factor=2  
    )

    return train_loader, val_loader, test_loader

def build_default_loaders(img_size,batch_size,split:str):
    match(split):
        case "test_training_set":
            return build_train_eval_loader(img_size,batch_size)
        case "test_test_set":
                _,_,test_dl = build_dataloaders(img_size=img_size,batch_size=batch_size)
                return test_dl

