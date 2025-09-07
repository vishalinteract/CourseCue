#!/usr/bin/env bash
set -e
python src/data_prep.py --dataset oulad --data_dir data/raw/oulad --out_dir data/processed
python src/train_eval.py --dataset oulad --model pop --k 10
python src/train_eval.py --dataset oulad --model bpr --epochs 3 --k 10
