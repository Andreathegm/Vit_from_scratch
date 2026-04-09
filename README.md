# Vit_from_scratch

A Vision Transformer (ViT-Tiny) trained from scratch on ImageNet-100, implemented in PyTorch.

---

## Project Structure

```
.
├── configs/
│   ├── model/          # Model architecture configs (Vit-Tiny.yaml, Vit-B.yaml)
│   ├── train/          # Training run configs
│   └── eval/           # Evaluation configs
├── data/
│   ├── dataset.py      # ImageDataset, TransformDataset, MixUpCutMixDataset
│   └── transforms.py   # Train / val / test transform pipelines
├── src/
│   ├── models/         # ViT, PatchEmbedding, Attention, MLP, TransformerEncoder
│   ├── engine/         # Training loop, evaluation, EarlyStopping, TrainSession
│   ├── utils/          # Metrics, checkpointing, factories, wandb logger, CSV manager
│   └── attention_rollout.py
├── main.py             # Entrypoint — train or test mode
├── plot_training_stats.py
└── run_all.sh          # Runs all evaluations sequentially
```

---

## Requirements

```bash
pip install torch torchvision
pip install wandb
pip install scikit-learn
pip install python-box pyyaml
pip install numpy matplotlib pandas
```

Python ≥ 3.10 is required (the codebase uses `match/case`).

---

## Dataset Setup

The project uses **ImageNet-100** (a 100-class subset of ImageNet).
<link_to_dataset>

Expected folder structure:

```
dataset/
└── imagenet-100/
    ├── train/          # One subfolder per class
    │   ├── n01440764/
    │   └── ...
    └── val.X/          # Test set — one subfolder per class
        ├── n01440764/
        └── ...
```

The train split is further divided internally: 90% training, 10% validation (stratified, seed = 42).

---

## Model

**ViT-Tiny** — a lightweight Vision Transformer with the following architecture:

| Hyperparameter | Value |
|----------------|-------|
| Image size | 224 × 224 |
| Patch size | 16 × 16 |
| Sequence length | 197 (196 patches + 1 CLS token) |
| Embedding dim | 192 |
| Depth | 12 blocks |
| Attention heads | 3 (head dim = 64) |
| MLP ratio | 4.0 |

?
| Dropout | 0.15 |
| Attention dropout | 0.10 |
?

| Parameters | ~5.5M |

Architecture config lives in `configs/model/Vit-Tiny.yaml`.

---

## Training

### Configuration

All training runs are driven by YAML config files in `configs/train/`. Key fields:

```yaml
model:
  name: "ViT-Tiny"
  # ... architecture params

training:
  run_name: "my-run"
  epochs: 150
  lr: 1e-4
  batch_size: 96
  weight_decay: 0.1
  scheduler: "cosine"          # "cosine" | "ReduceLROnPlateau"
  weights_path: null            # path to pretrained checkpoint, or null

early_stopping:
  patience: 25                  # null to disable

optimizer:
  name: "AdamW"
  weight_decay: 0.1
  lr: 1e-4

criterion:
  name: "CrossEntropyLoss"
  label_smoothing: 0.0

precision: "fp32"               # "fp16" for AMP
mixup_cutmix: true
```

### Run training

```bash
python main.py --config configs/train/<your_config>.yaml --mode train
```

Checkpoints are saved automatically under `checkpoints/<run_name>/`:
- `last.pt` — full state (model + optimizer + scheduler) after every epoch
- `best.pt` — best validation accuracy checkpoint

Training metrics are logged to **Weights & Biases** automatically. Set your API key before running:

```bash
wandb login
```

### Training pipeline

The runs build on each other sequentially:

| Run | Config | Description |
|-----|--------|-------------|
| 1 | `run1.yaml` | Fine-tune from pretrained weights, standard augmentation |
| 2 | `run2.yaml` | Fine-tune with MixUp + CutMix, stronger regularization |
| 3 | `run3finetuneofmixup.yaml` | Continue from run 2 at lower lr (`5e-5`) for 50 epochs |
| 4 | `train-vit-amp-mixup.yaml` | Mixed precision (fp16), ReduceLROnPlateau, early stopping |

Each run loads `weights_path` from the previous run's `best.pt`.

---

## Evaluation

### Configuration

Eval configs live in `configs/eval/`. Required fields:

```yaml
config_name: "my-run-eval"
model: ViT-Tiny
img_size: 224
batch_size: 100
criterion:
  name: CrossEntropyLoss
  label_smoothing: 0.0
weights_path: checkpoints/<run_name>/best.pt
split: test_test_set     # "test_test_set" | "test_training_set"
k: 5                     # Top-K for evaluation
```

### Run evaluation

```bash
python main.py --config configs/eval/<your_config>.yaml --mode test --csv csv_results/model_performance.csv
```

This prints Top-1 and Top-5 accuracy, saves per-class accuracy to `class_accuracy.npy`, and plots a bar chart to `class_accuracy.png`. Results are appended to the CSV if `--csv` is provided.

### Run all evaluations at once

```bash
bash run_all.sh
```

---

## Augmentation

The training transform pipeline (`data/transforms.py`) applies:

1. Resize to 240 × 240
2. Random crop to 224 × 224
3. Random horizontal flip
4. RandAugment (num\_ops=2, magnitude=9)
5. Normalize (ImageNet mean/std)
6. Random erasing (p=0.25)

MixUp/CutMix is applied on top via `MixUpCutMixDataset`:

| Parameter | Value |
|-----------|-------|
| MixUp alpha | 0.8 |
| CutMix alpha | 1.0 |
| Augmentation probability | 0.5 |
| Switch probability (CutMix vs MixUp) | 0.5 |

Labels are automatically converted to soft one-hot vectors when MixUp/CutMix is active.

---

## Reproducibility

The following seeds and settings are fixed for reproducibility:

- Train/val split: `sklearn.train_test_split` with `random_state=42`
- DataLoader: `shuffle=True` only for the training set

To fully reproduce a run:

1. Use the YAML config for that run.
2. Use the same checkpoint as starting point (`weights_path`).
3. Keep the dataset folder structure identical.
4. Match the Python, PyTorch, and torchvision versions.

Verified environment:

```
python      >= 3.10
torch       (with CUDA support recommended)
torchvision
scikit-learn
```

---

## Checkpoints

Checkpoints are excluded from the repository via `.gitignore`. Each checkpoint file is a dictionary with:

```python
{
    "model_state_dict":     ...,
    "optimizer_state_dict": ...,
    "scheduler_state_dict": ...,
    "epoch":                ...,
    "train_loss":           ...,
    "val_loss":             ...,
    "val_acc":              ...,
    "best_val_acc":         ...,
}
```

Use `load_weights_from_complex_checkpoint` for eval, or `TrainSession.resume()` to continue training.

---

## Attention Rollout

To visualize what the model attends to on a given image:

```python
from src.attention_rollout import get_imgs_attention_rollout

img_np, mask, img_masked = get_imgs_attention_rollout(
    model=model,
    image_path="path/to/image.jpg",
    device=device,
    patch_size=16,
    img_size=224
)
```

Use `src.utils.visualization.plot_single_rollout` or `plot_attention_grid` to display results.