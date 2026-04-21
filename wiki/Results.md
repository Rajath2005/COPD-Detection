# 📊 Results & Performance

## Table of Contents
1. [Evaluation Protocol](#evaluation-protocol)
2. [Target Performance](#target-performance)
3. [Metrics Explained](#metrics-explained)
4. [Expected Baseline CNN Results](#expected-baseline-cnn-results)
5. [Expected CNN-LSTM Results](#expected-cnn-lstm-results)
6. [Interpreting Output Files](#interpreting-output-files)
7. [Comparison with Related Work](#comparison-with-related-work)
8. [Limitations](#limitations)

---

## Evaluation Protocol

All models are evaluated on the **held-out ICBHI 2017 test set** (40% of recordings, patient-level split). Evaluation is performed only once for final reporting — hyperparameter tuning uses the validation set only.

| Split | % of data | Purpose |
|-------|-----------|---------|
| Training | 60% | Model fitting |
| Validation | 10% (from train) | Hyperparameter tuning, early stopping |
| Test | 40% | Final evaluation — used once |

---

## Target Performance

### Baseline CNN (Binary Classification)

| Metric | Target |
|--------|--------|
| Accuracy | > 90% |
| Sensitivity (Recall for COPD) | > 88% |
| Specificity | > 85% |
| F1-Score | > 89% |
| AUC-ROC | > 0.93 |

### Hybrid CNN-LSTM (4-class Severity)

| Metric | Target |
|--------|--------|
| Overall Accuracy | > 85% |
| Macro F1-Score | > 83% |
| Normal class F1 | > 90% |
| Severe class F1 | > 80% |

---

## Metrics Explained

### Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Proportion of all predictions that are correct. Can be misleading on imbalanced datasets — always review alongside F1.

### Precision

```
Precision = TP / (TP + FP)
```

Of all samples predicted as COPD, how many actually have COPD? High precision = few false alarms.

### Recall (Sensitivity)

```
Recall = TP / (TP + FN)
```

Of all actual COPD cases, how many did the model catch? **This is the most critical metric for a screening tool** — missing a COPD patient (false negative) is more dangerous than a false alarm.

### F1-Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Harmonic mean of precision and recall. Balances both concerns. Reported per-class and as a macro average.

### AUC-ROC

Area under the Receiver Operating Characteristic curve. Measures the model's ability to discriminate between classes at all classification thresholds. AUC = 1.0 is perfect; AUC = 0.5 is random.

### Specificity

```
Specificity = TN / (TN + FP)
```

True negative rate — proportion of healthy patients correctly identified as healthy.

---

## Expected Baseline CNN Results

> These are projected targets based on the literature; actual values will be reported after training.

### Binary Classification Report

```
              precision    recall  f1-score   support

      Normal       0.91      0.94      0.92      1256
        COPD       0.93      0.90      0.91      1108

    accuracy                           0.92      2364
   macro avg       0.92      0.92      0.92      2364
weighted avg       0.92      0.92      0.92      2364
```

### Confusion Matrix (Baseline CNN)

```
               Predicted
               Normal    COPD
Actual Normal  [ 1181      75 ]
       COPD    [  112     996 ]
```

---

## Expected CNN-LSTM Results

### 4-Class Classification Report

```
              precision    recall  f1-score   support

      Normal       0.94      0.96      0.95       841
        Mild       0.83      0.80      0.81       518
    Moderate       0.85      0.84      0.84       620
      Severe       0.88      0.87      0.87       385

    accuracy                           0.88      2364
   macro avg       0.88      0.87      0.87      2364
weighted avg       0.88      0.88      0.88      2364
```

### Confusion Matrix (CNN-LSTM)

```
                  Predicted
                  Normal   Mild   Mod   Severe
Actual Normal   [  807      24     8       2 ]
       Mild     [   38     415    54      11 ]
       Moderate [   12      52   521      35 ]
       Severe   [    4      12    34     335 ]
```

---

## Interpreting Output Files

After running `src/evaluate.py`, the `results/` directory contains:

### `metrics.json`

```json
{
  "model": "cnn_lstm",
  "test_accuracy": 0.882,
  "macro_f1": 0.872,
  "weighted_f1": 0.881,
  "auc_roc": {
    "Normal": 0.978,
    "Mild": 0.921,
    "Moderate": 0.934,
    "Severe": 0.956
  },
  "per_class": {
    "Normal":   {"precision": 0.94, "recall": 0.96, "f1": 0.95},
    "Mild":     {"precision": 0.83, "recall": 0.80, "f1": 0.81},
    "Moderate": {"precision": 0.85, "recall": 0.84, "f1": 0.84},
    "Severe":   {"precision": 0.88, "recall": 0.87, "f1": 0.87}
  }
}
```

### `confusion_matrix.png`

A normalised (row-wise) heatmap showing the fraction of each true class predicted as each class. The diagonal should be dark (correct predictions); off-diagonal cells reveal where the model confuses classes.

### `roc_curves.png`

One ROC curve per class (one-vs-rest), with AUC annotated. A good model will show curves close to the top-left corner.

### `loss_curve.png`

Training vs. validation loss and accuracy over epochs. Look for:
- **Converging lines** — good generalisation
- **Diverging lines** — overfitting; try more dropout or less capacity
- **Both lines high** — underfitting; try more capacity or longer training

---

## Comparison with Related Work

| Study | Model | Accuracy | F1 |
|-------|-------|---------|-----|
| Perna & Tagarelli (2019) | LSTM | 85.3% | 84.1% |
| Demir et al. (2020) | CNN | 87.4% | 86.8% |
| Ma et al. (2020) | BiLSTM+Attention | 86.1% | 85.3% |
| Nessrine et al. (2021) | CNN-LSTM | 89.2% | 88.7% |
| **This Work (target)** | **CNN-LSTM+Attention** | **> 90%** | **> 89%** |

---

## Limitations

| Limitation | Description |
|-----------|-------------|
| Dataset size | 920 recordings from 126 patients — relatively small for deep learning |
| Severity proxy | COPD severity is inferred from acoustic labels (crackle/wheeze) rather than spirometry-confirmed GOLD stages |
| Single dataset | All data from ICBHI 2017; cross-dataset generalisation is untested |
| Device variability | Performance may vary across microphone types not represented in training data |
| Paediatric bias | Some patients are children; adult and paediatric acoustics differ |
| No real-time testing | Pipeline is offline; real-time inference latency is uncharacterised |

These limitations will be discussed in detail in the final project report and journal paper.

---

*← [Model Training](Model-Training.md) | Next: [Roadmap →](Roadmap.md)*
