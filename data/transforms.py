import torchvision.transforms as T


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

_TRANSFORM_DISPATCHER = {
    "train": _get_train_transforms,
    "val": _get_eval_transforms,
    "test": _get_eval_transforms,
    "train.eval" : _get_eval_transforms
}

def get_transforms(img_size: int, trasform_name: str):
    if trasform_name not in _TRANSFORM_DISPATCHER:
        raise ValueError(f"Trasform {trasform_name} is not available"
                         f"Traformations available : {list(_TRANSFORM_DISPATCHER.keys())}")
    
    resize_size = int(img_size * (240 / 224))
    
    transform_factory = _TRANSFORM_DISPATCHER[trasform_name]
    return transform_factory(img_size, resize_size)