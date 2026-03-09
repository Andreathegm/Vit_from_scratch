from dataset import ImageDataset,TransformDataset,get_transforms
from torch.utils.data import DataLoader

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
        pin_memory=True,
        persistent_workers=True,  # evita di ricreare i worker ad ogni epoca
        prefetch_factor=2         # ogni worker prepara 2 batch in anticipo
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