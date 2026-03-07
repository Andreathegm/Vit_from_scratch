import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split


class TransformDataset(Dataset):
    """
    Wrapper che applica una transform indipendente a un Subset.
    Risolve il problema di train e val che condividono
    lo stesso ImageFolder sottostante.
    """
    def __init__(self, subset: Subset, transform, classes):
        self.subset = subset
        self.transform = transform
        self.classes = classes

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # subset[idx] restituisce (PIL Image, label)
        # perché ImageFolder non ha transform (è None)
        img, label = self.subset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __str__(self):
        return (
            f"TransformDataset("
            f"size={len(self)}, "
            f"classes={len(self.classes)})"
        )


class ImageDataset(Dataset):
    """
    Wrapper attorno a ImageFolder.
    La transform NON viene mai settata sull'ImageFolder —
    viene applicata solo nei TransformDataset figli.
    """
    def __init__(self, folder: str,transform=None):
        self.folder = folder
        # transform=None — le PIL Image vengono restituite grezze
        # la transform viene applicata solo in TransformDataset
        self.dataset = ImageFolder(root=folder, transform=transform)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __str__(self):
        return (
            f"ImageDataset("
            f"folder={self.folder}, "
            f"size={len(self)}, "
            f"classes={len(self.classes)})"
        )

    def split(self, val_ratio: float = 0.1):
        """
        Split stratificato in train e val.
        Restituisce due Subset — le transform vengono
        applicate dopo tramite TransformDataset.
        """
        targets = self.dataset.targets
        all_indices = list(range(len(self.dataset)))

        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=val_ratio,
            stratify=targets,
            random_state=42
        )

        train_subset = Subset(self.dataset, train_idx)
        val_subset   = Subset(self.dataset, val_idx)

        return train_subset, val_subset


def get_transforms(img_size: int, dataset_partition: str):
    resize_size = int(img_size * (240 / 224))

    match dataset_partition:

        case "train":
            return T.Compose([
                T.Resize(resize_size),
                T.RandomCrop(img_size),
                T.RandomHorizontalFlip(),
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                ),
                # RandAugment — DeiT paper, Touvron et al. 2021
                # num_ops=2: applica 2 trasformazioni casuali per immagine
                # magnitude=9: intensità delle trasformazioni
                T.RandAugment(num_ops=2, magnitude=9),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                # RandomErasing — cancella patch casuali dell'immagine
                # forza il modello a non dipendere da zone specifiche
                # Zhong et al. 2020 "Random Erasing Data Augmentation"
                T.RandomErasing(p=0.25, scale=(0.02, 0.33)),
            ])

        case "val" | "test":
            return T.Compose([
                T.Resize(resize_size),
                T.CenterCrop(img_size),   # più corretto di Resize diretto
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])