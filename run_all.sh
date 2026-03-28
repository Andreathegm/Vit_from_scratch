#!/bin/bash

# Stop script on error
set -e


#python main.py --config configs/eval/vit-finetune-mixuprun3.yaml --mode test --csv csv_results/model_performance.csv
python main.py --config configs/eval/vit-early-stopping.yaml --mode test --csv csv_results/model_performance.csv