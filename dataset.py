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


# def get_transforms(img_size: int, dataset_partition: str):
#     resize_size = int(img_size * (240 / 224))

#     match dataset_partition:

#         case "train":
#             return T.Compose([
#                 T.Resize(resize_size),
#                 T.RandomCrop(img_size),
#                 T.RandomHorizontalFlip(),
#                 # T.ColorJitter(
#                 #     brightness=0.4,
#                 #     contrast=0.4,
#                 #     saturation=0.4,
#                 #     hue=0.1
#                 # ),
#                 # RandAugment — DeiT paper, Touvron et al. 2021
#                 # num_ops=2: applica 2 trasformazioni casuali per immagine
#                 # magnitude=9: intensità delle trasformazioni
#                 T.RandAugment(num_ops=2, magnitude=9),
#                 T.ToTensor(),
#                 T.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 ),
#                 # RandomErasing — cancella patch casuali dell'immagine
#                 # forza il modello a non dipendere da zone specifiche
#                 # Zhong et al. 2020 "Random Erasing Data Augmentation"
#                 ## prima ho fatto con 0.33 
#                 T.RandomErasing(p=0.25, scale=(0.02, 0.10)),
#             ])

#         case "val" | "test":
#             return T.Compose([
#                 T.Resize(resize_size),
#                 T.CenterCrop(img_size),   # più corretto di Resize diretto
#                 T.ToTensor(),
#                 T.Normalize(
#                     mean=[0.485, 0.456, 0.406],
#                     std=[0.229, 0.224, 0.225]
#                 ),
#             ])

# 1. Estraiamo le costanti condivise per evitare duplicazioni (Principio DRY)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NORMALIZE = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

# 2. Definiamo funzioni specifiche per ogni pipeline
def _get_train_transforms(img_size: int, resize_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(resize_size),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        NORMALIZE,
        T.RandomErasing(p=0.25, scale=(0.02, 0.10)),
    ])

def _get_eval_transforms(img_size: int, resize_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(resize_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        NORMALIZE,
    ])

# 3. Creiamo il "Dispatcher" (Il nostro pattern)
_TRANSFORM_DISPATCHER = {
    "train": _get_train_transforms,
    "val": _get_eval_transforms,
    "test": _get_eval_transforms,
    "train.eval" : _get_eval_transforms
    # In futuro ti basterà aggiungere qui una riga:
    # "adv_train": _get_adversarial_train_transforms,
}

# 4. La funzione esposta diventa estremamente compatta
def get_transforms(img_size: int, dataset_partition: str):
    if dataset_partition not in _TRANSFORM_DISPATCHER:
        raise ValueError(f"Trasform config Available: '{dataset_partition}'. "
                         f"Valid : {list(_TRANSFORM_DISPATCHER.keys())}")
    
    resize_size = int(img_size * (240 / 224))
    
    # Recuperiamo la funzione dal dizionario e la eseguiamo passando i parametri
    transform_factory = _TRANSFORM_DISPATCHER[dataset_partition]
    return transform_factory(img_size, resize_size)

import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class MixUpCutMixDataset(Dataset):
    """
    Wraps a dataset and applies MixUp or CutMix augmentation on the fly.

    MixUp  — blends two images and their labels with a random weight.
             Zhang et al. 2018 "mixup: Beyond Empirical Risk Minimization"

    CutMix — cuts a random patch from one image and pastes it into another.
             Yun et al. 2019 "CutMix: Training Strategy that Makes Cut and Paste"

    Both techniques force the model to learn from ambiguous examples,
    reducing overfitting and improving generalization.
    Used in DeiT paper — Touvron et al. 2021, Table 9.
    """

    def __init__(
        self,
        dataset,
        num_classes:  int   = 100,
        mixup_alpha:  float = 0.8,  # beta distribution parameter for mixup
        cutmix_alpha: float = 1.0,  # beta distribution parameter for cutmix
        prob:         float = 0.5,  # probability of applying any augmentation
        switch_prob:  float = 0.5,  # probability of choosing cutmix over mixup
    ):
        self.dataset      = dataset
        self.num_classes  = num_classes
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob         = prob
        self.switch_prob  = switch_prob

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]
        # img1 shape: (C, H, W) — tensore float normalizzato
        # label1: intero scalare — indice della classe

        # con probabilità (1-prob) restituisce l'immagine originale
        # la label viene convertita in one-hot per compatibilità
        # con il resto del pipeline che si aspetta sempre label soft
        if np.random.random() > self.prob:
            return img1, self._one_hot(label1)

        # campiona un secondo esempio casuale dal dataset
        # può essere lo stesso idx — raro ma non è un problema
        idx2 = np.random.randint(len(self.dataset))
        img2, label2 = self.dataset[idx2]

        # sceglie quale augmentation applicare
        if np.random.random() < self.switch_prob:
            img, lam = self._mixup(img1, img2)
        else:
            img, lam = self._cutmix(img1, img2)

        # label morbida — combinazione convessa delle due label one-hot
        # lam=0.7, label1=gatto, label2=cane
        # → [0, ..., 0.7, ..., 0.3, ..., 0]
        #              gatto      cane
        label = lam * self._one_hot(label1) + (1 - lam) * self._one_hot(label2)

        return img, label

    def _one_hot(self, label) -> torch.Tensor:
        """Converts an integer label to a one-hot vector of size num_classes."""
        # gestisce sia label intere che già one-hot
        # utile se il dataset sottostante restituisce già label soft
        if isinstance(label, torch.Tensor) and label.dim() == 1:
            return label   # già one-hot — restituisce invariato
        t = torch.zeros(self.num_classes)
        t[label] = 1.0
        return t

    def _mixup(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        Blends two images pixel by pixel with weight lam.

        lam ~ Beta(alpha, alpha) — con alpha=0.8 lam è spesso vicino a 0 o 1
        quindi una delle due immagini domina — non sempre 50/50.

        img_out = lam * img1 + (1-lam) * img2
        """
        # campiona lam dalla distribuzione Beta
        # Beta(0.8, 0.8) produce valori concentrati vicino a 0 e 1
        # Beta(0.5, 0.5) produce valori più estremi
        # Beta(2.0, 2.0) produce valori più vicini a 0.5
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)

        # blend lineare pixel per pixel — stesso shape dell'input
        img = lam * img1 + (1 - lam) * img2
        # img shape: (C, H, W) — stessa shape di img1 e img2

        return img, lam

    def _cutmix(self, img1: torch.Tensor, img2: torch.Tensor):
        """
        Cuts a rectangular patch from img2 and pastes it into img1.

        The patch size is determined by lam:
          lam → piccolo  =  patch grande  (img2 domina)
          lam → grande   =  patch piccola (img1 domina)

        lam viene ricalcolato dopo il cut per riflettere
        l'area effettiva della patch — può differire dal lam iniziale
        a causa del clipping ai bordi dell'immagine.
        """
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)

        # H e W dall'immagine — dim 0 è il canale
        H, W = img1.shape[1], img1.shape[2]

        # dimensione del patch proporzionale a sqrt(1-lam)
        # sqrt perché l'area è proporzionale a h*w — due dimensioni
        cut_ratio = np.sqrt(1 - lam)
        cut_h     = int(H * cut_ratio)
        cut_w     = int(W * cut_ratio)

        # centro del patch — casuale dentro l'immagine
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        # coordinate della bounding box — clippate ai bordi
        x1 = max(cx - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W)
        y1 = max(cy - cut_h // 2, 0)
        y2 = min(cy + cut_h // 2, H)

        # copia img1 e incolla il patch di img2
        img = img1.clone()
        # [:, y1:y2, x1:x2] → tutti i canali, righe y1:y2, colonne x1:x2
        img[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]

        # ricalcola lam reale in base all'area effettiva del patch
        # necessario perché il clipping ai bordi può ridurre il patch
        lam = 1 - (x2 - x1) * (y2 - y1) / (H * W)

        return img, lam

def visualize_augmentations(dataset_base, mixup_cutmix_ds, n_samples: int = 10):
    """
    Plots original images alongside their MixUp and CutMix versions.
    Useful to verify the augmentation pipeline is working correctly.

    Args:
        dataset_base:      dataset senza MixUp/CutMix — immagini originali
        mixup_cutmix_ds:   MixUpCutMixDataset — immagini aumentate
        n_samples:         numero di esempi da visualizzare
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def denormalize(img: torch.Tensor) -> np.ndarray:
        """Inverts normalization for visualization."""
        img = img * std + mean              # denormalizza
        img = img.clamp(0, 1)              # clamp per evitare valori fuori range
        return img.permute(1, 2, 0).numpy()  # (C,H,W) → (H,W,C) per matplotlib

    fig, axes = plt.subplots(n_samples, 2, figsize=(8, n_samples * 3))
    fig.suptitle("Original  vs  MixUp/CutMix", fontsize=13, fontweight="bold")

    for i in range(n_samples):
        # immagine originale
        img_orig, label_orig = dataset_base[i]
        axes[i, 0].imshow(denormalize(img_orig))
        axes[i, 0].set_title(f"Original — class {label_orig}")
        axes[i, 0].axis("off")

        # immagine aumentata
        img_aug, label_soft = mixup_cutmix_ds[i]
        # classe dominante nella label soft
        dominant_class = label_soft.argmax().item()
        dominant_prob  = label_soft.max().item()
        axes[i, 1].imshow(denormalize(img_aug))
        axes[i, 1].set_title(
            f"Augmented — class {dominant_class} ({dominant_prob*100:.0f}%)"
        )
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("plots/mixup_cutmix_samples.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved → plots/mixup_cutmix_samples.png")