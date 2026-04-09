#!/bin/bash
# These names for the eval file (.yaml) are the orderded corrispective eval of the model obatined by Vit-pre-train,
# Vit-train-1,...,Vit-train-4
set -e

configs=(
    "configs/eval/vit-pretrain.yaml"
    "configs/eval/vit-finetune.yaml"
    "configs/eval/vit-finetune-mixup.yaml"
    "configs/eval/vit-finetune-mixuprun3.yaml"
    "configs/eval/vit-early-stopping.yaml"
)

for cfg in "${configs[@]}"; do
    python main.py --config "$cfg" --mode test --csv csv_results/model_performance.csv
done

