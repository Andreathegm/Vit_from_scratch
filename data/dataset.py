from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
import torch

class TransformDataset(Dataset):
    """
    Wrapper that applays a trasformation to a Subset.
    Solve the problem of incapabilty to applay different trasformations
    to two different subdatasets obtained from one ImageFolder obj(namely train and validation sets).
    """
    def __init__(self, subset: Subset, transform, classes):
        if subset.trasform is not None:
            raise RuntimeError(f"Subset {subset} must not have any trasformation")
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
            f"classes={len(self.classes)}"
            f"transform={self.transform})"
        )


class ImageDataset(Dataset):

    """
    Wrapper of ImageFolder.
    """
    def __init__(self, folder: str,transform=None):
        self.folder = folder
        self.dataset = ImageFolder(root=folder, transform=transform)
        self.classes = self.dataset.classes
        self.transform=transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __str__(self):
        return (
            f"ImageDataset("
            f"folder={self.folder}, "
            f"size={len(self)}, "
            f"classes={len(self.classes)}"
            f"transform={self.transform})"
        )

    def split(self, val_ratio: float = 0.1,seed = 42):
        targets = self.dataset.targets
        all_indices = list(range(len(self.dataset)))

        train_idx, val_idx = train_test_split(
            all_indices,
            test_size=val_ratio,
            stratify=targets,
            random_state=seed
        )

        train_subset = Subset(self.dataset, train_idx)
        val_subset   = Subset(self.dataset, val_idx)

        return train_subset, val_subset


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

        # with (1-prob) probability returns the original image
        # the label gets converted into one one-hot because the pipeline
        # expect softlabels instead of a single value
        if np.random.random() > self.prob:
            return img1, self._one_hot(label1)

        idx2 = np.random.randint(len(self.dataset))
        img2, label2 = self.dataset[idx2]

        # sceglie quale augmentation applicare
        if np.random.random() < self.switch_prob:
            img, lam = self._mixup(img1, img2)
        else:
            img, lam = self._cutmix(img1, img2)

        # convex combination of the two labels
        # lam=0.7, label1=cat, label2=dog
        #  [0, ..., 0.7, ..., 0.3, ..., 0]
        #           cat       dog
        label = lam * self._one_hot(label1) + (1 - lam) * self._one_hot(label2)

        return img, label

    def _one_hot(self, label: torch.Tensor | int) -> torch.Tensor:
        """Converts an integer label or a one-hot vector to a one-hot vector of size num_classes."""
        
        # Caso 1: è già un tensore 1D → controlliamo che sia davvero one-hot
        if isinstance(label, torch.Tensor) and label.dim() == 1:
            if label.numel() != self.num_classes:
                raise ValueError(f"One-hot vector has wrong length: expected {self.num_classes}, got {label.numel()}")
            if not torch.all((label == 0) | (label == 1)):
                raise ValueError("Vector contains values other than 0 or 1")
            if label.sum().item() != 1:
                raise ValueError("Vector is not one-hot (must contain exactly one '1')")
            return label
        
        # Caso 2: è un intero → convertiamo in one-hot
        if isinstance(label, int):
            if not (0 <= label < self.num_classes):
                raise ValueError(f"Label {label} out of range [0, {self.num_classes-1}]")
            t = torch.zeros(self.num_classes)
            t[label] = 1.0
            return t
    
        raise TypeError("Label must be either an integer or a 1D tensor")


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