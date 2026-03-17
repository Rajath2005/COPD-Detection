# 🫁 COPD-Detection

<div align="center">

![Banner](https://capsule-render.vercel.app/api?type=waving&color=0:0D1117,50:0a1628,100:0d2137&height=200&section=header&text=COPD%20Detection&fontSize=48&fontColor=00FFD1&animation=fadeIn&fontAlignY=38&desc=Deep%20Learning%20•%20Respiratory%20Sound%20Analysis%20•%20CNN-LSTM&descSize=16&descColor=7FDBFF&descAlignY=58)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)
[![VCET](https://img.shields.io/badge/VCET_Puttur-VTU_Affiliated-0D1117?style=for-the-badge&labelColor=0a1628)](https://vcetputtur.ac.in)
[![Status](https://img.shields.io/badge/Status-Phase_I_Active-00FFD1?style=for-the-badge&labelColor=0D1117)](https://github.com/Rajath2005/COPD-Detection)

> **Deep Learning-Based Early COPD Detection and Severity Classification Using Respiratory Sound Analysis**  
> VTU Major Project | Department of Computer Science | VCET Puttur | 2025–26

</div>

---

## 📌 Overview

Chronic Obstructive Pulmonary Disease (COPD) is a leading cause of morbidity worldwide, often diagnosed late due to lack of accessible screening tools. This project builds an AI-powered, non-invasive screening system that analyzes **respiratory sounds** to detect COPD early and classify its severity.

We use the **ICBHI 2017 Respiratory Sound Database** and a hybrid **CNN-LSTM** deep learning architecture to capture both spatial and temporal features from audio signals.

---

## 👥 Team

| USN | Name | Role |
|---|---|---|
| 4VP23CS084 | Sanath K | Team Leader |
| 4VP23CS070 | Rajath Kiran A | Team Member |
| 4VP23CS076 | Rithesh | Team Member |
| 4VP23CS093 | Sheethal D Rai | Team Member |

**Guide:** Prof. Pramod Kumar PM, Department of CS, VCET Puttur

---

## 🎯 Objectives

- Collect and preprocess the ICBHI 2017 respiratory sound dataset (noise reduction, segmentation, MFCC/Mel-spectrogram extraction)
- Develop a **baseline CNN model** for binary COPD classification (present / absent)
- Build an **enhanced CNN-LSTM hybrid** model for 4-class severity classification: Normal → Mild → Moderate → Severe
- Validate using accuracy, precision, recall, and F1-score

---

## 🧠 Methodology

```
Respiratory Sound Input
        │
        ▼
  Data Preprocessing
  (Noise reduction, Resampling, Segmentation)
        │
        ▼
  Feature Extraction
  (Mel-spectrogram / MFCC)
        │
        ▼
  Deep Learning Model
  (Baseline CNN → Hybrid CNN-LSTM)
        │
        ▼
  COPD Detection (Binary)
        │
        ▼
  Severity Classification
  (Normal / Mild / Moderate / Severe)
        │
        ▼
  Performance Evaluation
  (Accuracy, Precision, Recall, F1)
        │
        ▼
  COPD Risk Result Output
```

---

## 🗂️ Repository Structure

```
COPD-Detection/
│
├── data/
│   ├── raw/                  # Original ICBHI 2017 audio files
│   ├── processed/            # Cleaned, segmented audio
│   └── features/             # Extracted MFCC / Mel-spectrogram arrays
│
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb
│   ├── 03_baseline_cnn.ipynb
│   └── 04_cnn_lstm_hybrid.ipynb
│
├── src/
│   ├── preprocess.py         # Preprocessing pipeline
│   ├── features.py           # Feature extraction (MFCC, Mel)
│   ├── models/
│   │   ├── baseline_cnn.py
│   │   └── cnn_lstm.py
│   ├── train.py
│   └── evaluate.py
│
├── results/
│   ├── plots/                # Confusion matrices, loss curves
│   └── metrics.json          # Saved evaluation results
│
├── reports/
│   ├── synopsis.pdf
│   └── progress_reports/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| Deep Learning | PyTorch / TensorFlow |
| Audio Processing | Librosa, SoundFile |
| Data & Visualization | NumPy, Pandas, Matplotlib, Seaborn |
| Notebooks | Jupyter |
| Dataset | ICBHI 2017 Respiratory Sound Database |

---

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/Rajath2005/COPD-Detection.git
cd COPD-Detection

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download ICBHI 2017 dataset
# → https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
# Place audio files in data/raw/

# 5. Run preprocessing
python src/preprocess.py

# 6. Launch notebooks
jupyter notebook notebooks/
```

---

## 📅 Project Timeline

| Milestone | Deadline | Status |
|---|---|---|
| Synopsis submission | 27 Feb 2026 | ✅ Done |
| Synopsis presentation | 03–10 Mar 2026 | ✅ Done |
| Progress Reports 1 & 2 (Intro + Requirements) | 14 Apr 2026 | 🔄 In Progress |
| Presentation 1 | 2nd Week May 2026 | ⏳ Upcoming |
| Progress Reports 3 & 4 (Design + Implementation) | 4th Week Aug 2026 | ⏳ Upcoming |
| Presentation 2 | 2nd Week Sep 2026 | ⏳ Upcoming |
| Progress Reports 5 & 6 (Testing + Results) | 2nd Week Oct 2026 | ⏳ Upcoming |
| Final Report + Journal Paper | 20 Nov 2026 | ⏳ Upcoming |
| Internal Viva / Project Exhibition | 4th Week Nov 2026 | ⏳ Upcoming |

---

## 📊 Expected Results

- Binary classification (COPD / Normal) with target accuracy **> 90%**
- 4-class severity model (Normal / Mild / Moderate / Severe)
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

---

## 🏥 Applications

- **Early COPD Screening** — detect COPD before severe symptoms develop
- **Remote Healthcare** — low-cost solution for rural/resource-limited areas
- **AI Clinical Assistance** — reduce manual effort for healthcare professionals
- **Research & Training** — study respiratory acoustic patterns with AI

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- **ICBHI 2017 Challenge** for the respiratory sound dataset
- **Prof. Pramod Kumar PM** for guidance and mentorship
- **VCET Puttur, VTU** for academic support

---

<div align="center">

![Footer](https://capsule-render.vercel.app/api?type=waving&color=0:0d2137,100:0D1117&height=100&section=footer)

*Built with 🫁 + 💻 by Team COPD-Detection, VCET Puttur*

</div>
