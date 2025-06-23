#!/bin/bash

# Exit on error
set -e

# Activate conda env (if you want)
conda activate health-risk-prediction

echo "Training model..."
python src/train_model.py

echo "Evaluating model (target recall = 0.9)..."
python src/evaluate_model.py --target_recall 0.9

echo "Done! Latest results are saved to:"
echo "- models/"
echo "- reports/evaluation_report.md"
echo "- README.md (latest results updated)"
