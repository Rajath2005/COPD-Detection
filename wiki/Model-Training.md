# 🏋️ Model Training Guide

## Table of Contents
1. [Training Configuration](#training-configuration)
2. [Data Preparation Checklist](#data-preparation-checklist)
3. [Baseline CNN Training](#baseline-cnn-training)
4. [Hybrid CNN-LSTM Training](#hybrid-cnn-lstm-training)
5. [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Monitoring Training](#monitoring-training)
7. [Resuming Training](#resuming-training)
8. [Evaluation](#evaluation)
9. [Model Comparison](#model-comparison)
10. [Best Practices](#best-practices)

---

## Training Configuration

Training is controlled via command-line arguments or an optional JSON config file:

```bash
python src/train.py --config configs/cnn_lstm_config.json
```

Example config (`configs/cnn_lstm_config.json`):

```json
{
  "model": "cnn_lstm",
  "features_dir": "data/features",
  "epochs": 150,
  "batch_size": 16,
  "lr": 0.0005,
  "num_classes": 4,
  "lstm_hidden": 128,
  "lstm_layers": 2,
  "attention": true,
  "bidirectional": true,
  "weight_decay": 1e-4,
  "lr_scheduler": "cosine",
  "early_stop_patience": 15,
  "seed": 42,
  "out_dir": "models/cnn_lstm"
}
```

---

## Data Preparation Checklist

Before training, verify the following:

- [ ] Raw ICBHI 2017 audio files extracted to `data/raw/`
- [ ] `python src/preprocess.py` completed without errors
- [ ] `python src/features.py` completed and `data/features/` contains `.npy` files
- [ ] `data/features/X_train.npy` shape is `(N, 3, 128, 128)`
- [ ] Class labels in `data/features/y_train.npy` are integers 0–3 (4-class) or 0–1 (binary)

Quick check:

```bash
python - <<'EOF'
import numpy as np
X = np.load("data/features/X_train.npy")
y = np.load("data/features/y_train.npy")
print(f"X_train shape: {X.shape}")
print(f"y_train unique classes: {np.unique(y, return_counts=True)}")
EOF
```

---

## Baseline CNN Training

### Recommended settings

```bash
python src/train.py \
  --model baseline_cnn \
  --features-dir data/features \
  --num-classes 2 \
  --epochs 100 \
  --batch-size 32 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --lr-scheduler plateau \
  --early-stop-patience 10 \
  --seed 42 \
  --out-dir models/baseline_cnn
```

### Expected training behaviour

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|---------|-------------|
| 10 | ~0.55 | ~0.62 | ~72% |
| 30 | ~0.35 | ~0.44 | ~80% |
| 60 | ~0.22 | ~0.36 | ~85% |
| 100 | ~0.18 | ~0.32 | ~88% |

*(values are approximate — will vary with data split and augmentation)*

---

## Hybrid CNN-LSTM Training

### Recommended settings

```bash
python src/train.py \
  --model cnn_lstm \
  --features-dir data/features \
  --num-classes 4 \
  --epochs 150 \
  --batch-size 16 \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --lstm-hidden 128 \
  --lstm-layers 2 \
  --attention \
  --bidirectional \
  --lr-scheduler cosine \
  --early-stop-patience 15 \
  --seed 42 \
  --out-dir models/cnn_lstm
```

### Expected training behaviour

| Epoch | Train Loss | Val Loss | Val Accuracy |
|-------|-----------|---------|-------------|
| 20 | ~1.20 | ~1.35 | ~55% |
| 50 | ~0.80 | ~0.95 | ~70% |
| 100 | ~0.50 | ~0.68 | ~82% |
| 150 | ~0.35 | ~0.58 | ~88% |

---

## Hyperparameter Tuning

### Key hyperparameters to tune

| Hyperparameter | Search Range | Impact |
|----------------|-------------|--------|
| Learning rate | [1e-4, 1e-2] | High |
| Batch size | [8, 16, 32, 64] | Medium |
| LSTM hidden units | [64, 128, 256] | High |
| LSTM layers | [1, 2, 3] | Medium |
| Dropout rate | [0.2, 0.5] | Medium |
| CNN filters | [32/64/128, 64/128/256] | Medium |
| Weight decay | [1e-5, 1e-3] | Low–Medium |

### Strategy

1. **Start coarse** — grid search over {lr, batch_size} with fixed architecture
2. **Architecture search** — tune LSTM units and CNN filter counts
3. **Regularisation** — tune dropout and weight decay to reduce overfitting
4. **Fine-grained LR** — cosine annealing with warm restarts for final tuning

Manual hyperparameter tuning is recommended for this project scale. If compute permits, use `optuna` for automated search:

```bash
pip install optuna
python src/tune.py --model cnn_lstm --n-trials 50
```

---

## Monitoring Training

Training progress is printed to stdout:

```
Epoch [1/150]  Train Loss: 1.384  Val Loss: 1.401  Val Acc: 48.2%  LR: 5.00e-04
Epoch [2/150]  Train Loss: 1.201  Val Loss: 1.318  Val Acc: 52.7%  LR: 5.00e-04
...
✅ Best model saved at epoch 87  (Val Acc: 88.1%)
```

Additionally, training metrics are logged to `models/<name>/training_log.csv` for later plotting:

```bash
python - <<'EOF'
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("models/cnn_lstm/training_log.csv")
plt.plot(df["epoch"], df["val_accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("CNN-LSTM Training")
plt.savefig("results/cnn_lstm/loss_curve.png")
EOF
```

---

## Resuming Training

If training is interrupted, resume from the last checkpoint:

```bash
python src/train.py \
  --model cnn_lstm \
  --features-dir data/features \
  --resume models/cnn_lstm/last_model.pth \
  --epochs 150 \
  --out-dir models/cnn_lstm
```

The script automatically detects the epoch at which training stopped and continues.

---

## Evaluation

After training, run evaluation on the held-out test set:

```bash
python src/evaluate.py \
  --model-path models/cnn_lstm/best_model.pth \
  --model-type cnn_lstm \
  --features-dir data/features \
  --num-classes 4 \
  --out-dir results/cnn_lstm
```

Saves:
- `metrics.json` — accuracy, precision, recall, F1, AUC per class
- `confusion_matrix.png`
- `roc_curves.png`
- `classification_report.txt`

---

## Model Comparison

After training both models, compare their performance:

```bash
python src/compare_models.py \
  --results-dirs results/baseline_cnn results/cnn_lstm \
  --out-dir results/comparison
```

This generates a side-by-side comparison table and bar chart of all metrics.

---

## Best Practices

1. **Always set a random seed** (`--seed 42`) for reproducibility
2. **Use class weights** — the dataset is imbalanced; weighted cross-entropy significantly improves minority-class recall
3. **Train on GPU when possible** — CNN-LSTM training takes ~30 min on GPU vs. ~4 hours on CPU
4. **Monitor validation loss, not training loss** — training loss will always decrease; validation loss reveals overfitting
5. **Do not evaluate on the test set during hyperparameter tuning** — use only the validation set; evaluate on test only for final reporting
6. **Save best checkpoint by validation loss** — not by training loss or validation accuracy
7. **Use early stopping** — prevents overfitting without manual epoch monitoring

---

*← [Usage](Usage.md) | Next: [Results →](Results.md)*
