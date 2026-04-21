# ⚙️ Installation & Setup

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Clone the Repository](#clone-the-repository)
3. [Create a Virtual Environment](#create-a-virtual-environment)
4. [Install Dependencies](#install-dependencies)
5. [Download the Dataset](#download-the-dataset)
6. [Verify Installation](#verify-installation)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Ensure the following are installed on your system before setting up the project:

| Requirement | Minimum Version | Recommended |
|-------------|----------------|-------------|
| Python | 3.10 | 3.11 |
| pip | 22.x | latest |
| Git | 2.x | latest |
| CUDA (optional) | 11.8 | 12.x (for GPU training) |
| RAM | 8 GB | 16+ GB |
| Disk space | 5 GB (data + model) | 10 GB |

> **GPU Note:** Training is significantly faster with a CUDA-capable GPU. The code falls back to CPU automatically if no GPU is detected.

---

## Clone the Repository

```bash
git clone https://github.com/Rajath2005/COPD-Detection.git
cd COPD-Detection
```

---

## Create a Virtual Environment

It is strongly recommended to use a dedicated virtual environment to avoid dependency conflicts.

### Using `venv` (standard library)

```bash
# Create the environment
python -m venv venv

# Activate — Linux / macOS
source venv/bin/activate

# Activate — Windows (Command Prompt)
venv\Scripts\activate.bat

# Activate — Windows (PowerShell)
venv\Scripts\Activate.ps1
```

### Using `conda` (alternative)

```bash
conda create -n copd-detection python=3.11
conda activate copd-detection
```

---

## Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Core Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchaudio` | Deep learning framework and audio utilities |
| `tensorflow` (optional alt.) | Alternative deep learning framework |
| `librosa` | Audio loading, resampling, feature extraction |
| `soundfile` | Reading/writing audio files |
| `numpy` | Numerical computing |
| `pandas` | Data handling and annotation parsing |
| `scikit-learn` | Metrics, splitting, class weighting |
| `matplotlib` | Plotting loss curves, confusion matrices |
| `seaborn` | Enhanced visualisations |
| `jupyter` | Interactive notebooks |
| `audiomentations` | Audio data augmentation |
| `tqdm` | Progress bars |

### GPU-Specific Installation (PyTorch + CUDA 12.x)

If you have a compatible GPU, install the CUDA-enabled PyTorch build:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Download the Dataset

1. Visit the ICBHI 2017 dataset page:  
   👉 [https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)

2. Download `ICBHI_final_database.zip` (~1.2 GB)

3. Extract into the `data/raw/` directory:

```bash
# Example on Linux/macOS
unzip ICBHI_final_database.zip -d data/raw/
```

Your directory structure should look like:

```
data/
└── raw/
    ├── 101_1b1_Al_sc_Meditron.wav
    ├── 101_1b1_Al_sc_Meditron.txt
    ├── 102_1b1_Ar_sc_Meditron.wav
    ├── ...
    └── patient_diagnosis.csv
```

> **Note:** The `data/` directory is in `.gitignore` — do not commit audio data to version control.

---

## Verify Installation

Run the following check to verify all core libraries are installed correctly:

```bash
python - <<'EOF'
import torch
import librosa
import numpy as np
import pandas as pd
import sklearn
import matplotlib

print(f"PyTorch      : {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Librosa      : {librosa.__version__}")
print(f"NumPy        : {np.__version__}")
print(f"Pandas       : {pd.__version__}")
print(f"Scikit-learn : {sklearn.__version__}")
print(f"Matplotlib   : {matplotlib.__version__}")
print("\n✅ All dependencies installed successfully.")
EOF
```

Expected output:

```
PyTorch      : 2.x.x
CUDA available: True   ← False if no GPU
Librosa      : 0.10.x
NumPy        : 1.26.x
...
✅ All dependencies installed successfully.
```

---

## Troubleshooting

### `librosa` import error

```bash
pip install --upgrade librosa soundfile numba
```

### `torch` not finding CUDA

Check your CUDA version:
```bash
nvidia-smi
```
Then install the matching PyTorch CUDA build from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/).

### Memory errors during preprocessing

- Reduce the batch size in `src/preprocess.py`
- Process in smaller chunks by setting `--chunk-size` argument

### Jupyter notebook kernel not found

```bash
pip install ipykernel
python -m ipykernel install --user --name copd-detection
```
Then select `copd-detection` kernel in Jupyter.

### Permission denied on `venv/Scripts/Activate.ps1` (Windows)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

*← [Dataset](Dataset.md) | Next: [Usage →](Usage.md)*
