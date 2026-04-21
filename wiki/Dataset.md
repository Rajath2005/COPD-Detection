# 🗃️ Dataset — ICBHI 2017 Respiratory Sound Database

## Table of Contents
1. [Overview](#overview)
2. [Dataset Statistics](#dataset-statistics)
3. [Recording Equipment](#recording-equipment)
4. [Class Distribution](#class-distribution)
5. [File Format & Annotations](#file-format--annotations)
6. [Downloading the Dataset](#downloading-the-dataset)
7. [Dataset Preprocessing Notes](#dataset-preprocessing-notes)
8. [Data Splits](#data-splits)
9. [Ethical Considerations](#ethical-considerations)

---

## Overview

The **ICBHI 2017 Respiratory Sound Database** is the benchmark dataset used in this project. It was released as part of the **International Conference on Biomedical and Health Informatics (ICBHI) 2017 Scientific Challenge** by researchers from Portugal, Greece, and the UK.

It is the largest publicly available, annotated respiratory sound dataset, widely used in the respiratory AI research community.

- **Full name:** ICBHI 2017 Respiratory Sound Database
- **Download:** [https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
- **License:** Creative Commons Attribution (CC BY)
- **Paper:** Rocha et al., "An open access database for the evaluation of respiratory sound classification algorithms", *Physiological Measurement*, 2019

---

## Dataset Statistics

| Property | Value |
|----------|-------|
| Total recordings | 920 WAV audio files |
| Total breathing cycles | 6,898 |
| Total patients | 126 |
| Total annotated events | 6,898 cycles |
| Recording duration range | 10–90 seconds per file |
| Sample rates | 4,000 Hz / 10,000 Hz / 44,100 Hz (mixed) |
| Bit depth | 16-bit PCM |
| Chest locations recorded | 7 (trachea, anterior/posterior left/right, lateral left/right) |
| Recording modes | Sequential, simultaneous |
| Patient age range | Paediatric to elderly |

---

## Recording Equipment

| Device | Stethoscope Type | Used For |
|--------|-----------------|----------|
| AKG C417L | Mic + electronic stethoscope | Adult recordings |
| 3M Littmann Classic II SE | Acoustic stethoscope | General use |
| 3M Littmann 3200 | Electronic stethoscope | General use |
| WelchAllyn Meditron | Electronic stethoscope | Clinical trials |

Recordings come from multiple clinical sites, creating realistic device and environment variability.

---

## Class Distribution

### Breathing Cycle Labels

Each breathing cycle is annotated with the presence or absence of two pathological sounds:

| Sound Event | Description |
|-------------|-------------|
| **Crackle** | Discontinuous, explosive adventitious sound — associated with fibrosis, pneumonia, COPD |
| **Wheeze** | Continuous, musical adventitious sound — associated with asthma, COPD, bronchitis |

| Class | Cycles | % of total |
|-------|--------|-----------|
| Normal (no crackle, no wheeze) | 3,642 | 52.8% |
| Crackle only | 1,864 | 27.0% |
| Wheeze only | 886 | 12.8% |
| Both Crackle and Wheeze | 506 | 7.3% |

### Patient Diagnosis Labels

| Diagnosis | Patients | Description |
|-----------|---------|-------------|
| COPD | 64 | Confirmed COPD diagnosis |
| Healthy | 26 | No respiratory disease |
| Asthma | 7 | Asthma diagnosis |
| URTI | 14 | Upper Respiratory Tract Infection |
| Bronchiectasis | 7 | Bronchiectasis |
| Pneumonia | 6 | Pneumonia |
| LRTI | 2 | Lower Respiratory Tract Infection |

For this project, we map patient diagnoses to the 4-class COPD severity scheme using the GOLD staging criteria (where available) and the audio annotation labels as a proxy.

---

## File Format & Annotations

### Audio Files

Files are named following the convention:

```
{patient_ID}_{recording_index}_{chest_location}_{acquisition_mode}_{recording_equipment}.wav
```

Example: `101_1b1_Al_sc_Meditron.wav`

- `101` → Patient ID
- `1b1` → Recording index
- `Al` → Anterior left chest location
- `sc` → Sequential, single channel
- `Meditron` → WelchAllyn Meditron stethoscope

### Annotation Files

Each WAV file has a corresponding `.txt` annotation file with tab-separated rows:

```
<cycle_start_seconds> <cycle_end_seconds> <crackle_flag> <wheeze_flag>
```

Example:
```
0.036   1.018   0   0
1.018   1.890   1   0
1.890   2.900   0   1
```

### Patient Metadata

A separate `patient_diagnosis.txt` CSV-style file maps each patient ID to their clinical diagnosis.

---

## Downloading the Dataset

1. Visit [https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge](https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge)
2. Register / agree to the data usage terms
3. Download the ZIP archive (`ICBHI_final_database.zip`, ~1.2 GB)
4. Extract into the `data/raw/` directory of this project:

```
data/
└── raw/
    ├── 101_1b1_Al_sc_Meditron.wav
    ├── 101_1b1_Al_sc_Meditron.txt
    ├── ...
    └── patient_diagnosis.csv
```

> **Note:** Do not commit the raw dataset to this repository. The `data/raw/` directory is included in `.gitignore`.

---

## Dataset Preprocessing Notes

| Step | Tool | Detail |
|------|------|--------|
| Resample to 22,050 Hz | `librosa.resample` | Unifies sample rates across devices |
| Bandpass filter 100–8,000 Hz | `scipy.signal.butter` | Removes non-respiratory noise |
| Amplitude normalisation | peak scaling | Removes volume differences across devices |
| Cycle extraction | annotation timestamps | Creates individual cycle audio clips |
| Feature extraction | `librosa` | MFCC (40 coeff) + Mel-spectrogram (128 mel bands) |
| Zero-padding / truncation | numpy | Standardise to 128 time frames |
| Data augmentation | custom + `audiomentations` | Time stretch, pitch shift, noise, SpecAugment |

See `src/preprocess.py` and `src/features.py` for implementation details.

---

## Data Splits

The ICBHI challenge defines an official 60/40 train/test split at the recording level (not the cycle level) to prevent patient-level data leakage:

| Split | Recordings | Cycles (approx.) |
|-------|-----------|-----------------|
| Training | 60% (552) | ~4,100 |
| Validation | 10% (split from training) | ~680 |
| Test | 40% (368) | ~2,100 |

> **Important:** Splits are made at the **patient level**, so no cycles from the same patient appear in both train and test sets.

---

## Ethical Considerations

- All data was collected with patient informed consent according to the ethics committees of the contributing institutions
- No personally identifiable information (PII) is included in the dataset; patients are identified only by anonymous numeric IDs
- This project is academic and non-commercial; data is used strictly in accordance with the CC BY licence terms

---

*← [Architecture](Architecture.md) | Next: [Installation →](Installation.md)*
