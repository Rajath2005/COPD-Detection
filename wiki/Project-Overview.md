# 📌 Project Overview

## Table of Contents
1. [Background](#background)
2. [Problem Statement](#problem-statement)
3. [Motivation](#motivation)
4. [Objectives](#objectives)
5. [Scope](#scope)
6. [Significance](#significance)
7. [Related Work](#related-work)

---

## Background

**Chronic Obstructive Pulmonary Disease (COPD)** is a group of chronic inflammatory lung diseases — primarily emphysema and chronic bronchitis — that cause obstructed airflow from the lungs. It is characterised by:

- Persistent respiratory symptoms (dyspnoea, chronic cough, sputum production)
- Progressive and largely irreversible airflow limitation
- Accelerated decline in lung function over time

### Epidemiology

| Statistic | Value |
|-----------|-------|
| Global prevalence | ~300 million people |
| Annual deaths worldwide | ~3.2 million (3rd leading cause of death) |
| India prevalence | ~55 million |
| Diagnosed on time | < 30% of cases |
| Primary risk factor | Tobacco smoking (70–80% of cases); also air pollution, occupational dust |

### COPD Severity Stages (GOLD Classification)

| Stage | FEV₁ % Predicted | Description |
|-------|-------------------|-------------|
| I — Mild | ≥ 80% | Minimal symptoms; often undiagnosed |
| II — Moderate | 50–79% | Breathlessness on exertion; first medical visit |
| III — Severe | 30–49% | Frequent exacerbations; significant limitation |
| IV — Very Severe | < 30% | Life-threatening; chronic respiratory failure |

---

## Problem Statement

Current COPD diagnosis relies on:
- **Spirometry** — requires trained technicians and calibrated equipment
- **CT chest scans** — expensive and exposes patients to radiation
- **Clinical evaluation** — subjective and available only in urban hospitals

These barriers result in **late-stage diagnosis**, by which time irreversible lung damage has already occurred. There is a critical unmet need for a **low-cost, non-invasive, portable COPD screening tool** that can be used in primary care and remote settings.

---

## Motivation

Respiratory sounds — specifically **crackles** and **wheezes** — are well-established acoustic biomarkers of obstructive and restrictive lung disease. Advances in:

- Digital stethoscopes and portable microphones
- Deep learning models (CNNs, LSTMs, Transformers)
- Standardised audio datasets (ICBHI 2017)

…make it feasible to build AI systems that automatically detect abnormal lung sounds with clinician-level performance.

This project was motivated by the opportunity to bridge the gap between AI research and practical clinical deployment in resource-constrained environments.

---

## Objectives

1. **Data Acquisition & Preprocessing**
   - Obtain the ICBHI 2017 Respiratory Sound Database
   - Apply noise reduction, signal normalisation, and cycle segmentation
   - Extract Mel-Frequency Cepstral Coefficients (MFCCs) and Mel-spectrograms

2. **Baseline Model Development**
   - Build a Convolutional Neural Network (CNN) for binary COPD classification (COPD present / absent)
   - Establish performance benchmarks

3. **Hybrid CNN-LSTM Model**
   - Design an enhanced CNN-LSTM architecture to capture both spatial (spectral) and temporal (sequential) patterns in audio
   - Perform 4-class severity classification: Normal → Mild → Moderate → Severe

4. **Evaluation & Validation**
   - Measure accuracy, precision, recall, and F1-score
   - Generate confusion matrices and ROC curves
   - Compare baseline vs. hybrid model performance

5. **Documentation & Reporting**
   - Maintain progress reports aligned with VTU Major Project requirements
   - Produce a journal-quality report of findings

---

## Scope

### In Scope
- Audio classification using the ICBHI 2017 dataset
- Binary COPD detection (Yes / No)
- 4-class COPD severity grading
- CNN and CNN-LSTM deep learning models
- Python-based implementation with PyTorch / TensorFlow
- Standard evaluation metrics

### Out of Scope
- Real-time clinical deployment or FDA/CE certification
- Multi-modal data (e.g., combining audio with imaging or blood tests)
- Patient longitudinal tracking
- Mobile or edge device optimisation (potential future phase)

---

## Significance

| Stakeholder | Benefit |
|-------------|---------|
| Patients | Earlier detection → better outcomes and quality of life |
| Rural / underserved communities | Access to AI screening without specialist infrastructure |
| Healthcare providers | AI-assisted triage reduces workload and missed diagnoses |
| Researchers | Reproducible deep learning benchmark for respiratory AI |
| Public health | Scalable screening tool to reduce COPD mortality |

---

## Related Work

| Study | Approach | Dataset | Accuracy |
|-------|----------|---------|----------|
| Perna & Tagarelli (2019) | LSTM on MFCCs | ICBHI 2017 | 85.3% |
| Demir et al. (2020) | CNN on Mel-spectrograms | ICBHI 2017 | 87.4% |
| Ma et al. (2020) | BiLSTM + Attention | ICBHI 2017 | 86.1% |
| Nessrine et al. (2021) | CNN-LSTM Hybrid | ICBHI 2017 | 89.2% |
| **This Work** | CNN-LSTM Hybrid (enhanced) | ICBHI 2017 | **> 90% (target)** |

This project builds on existing work by combining improved preprocessing (adaptive noise filtering, cycle-level augmentation) with a refined CNN-LSTM architecture and systematic hyperparameter tuning.

---

*← [Home](Home.md) | Next: [Architecture →](Architecture.md)*
