"""
features.py — Feature Extraction Pipeline
==========================================
Reads cycle WAV files from data/processed/, extracts:
  - Mel-spectrogram  (128 mel bands)
  - MFCC             (40 coefficients)
  - Delta-MFCC       (first-order delta of MFCC)

Stacks them into a 3-channel tensor of shape (3, 128, 128),
applies train-time data augmentation (optional),
splits into train/val/test sets at the PATIENT level,
and saves numpy arrays to data/features/.

Usage (PowerShell):
    python src/features.py
    python src/features.py --input-dir data/processed --out-dir data/features --augment

Author: VCET COPD-Detection Team
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Constants  (must match Architecture.md spec)
# ──────────────────────────────────────────────
TARGET_SR   = 22_050
N_MELS      = 128
N_MFCC      = 40
N_FFT       = 1024
HOP_LENGTH  = 256
MAX_FRAMES  = 128          # all features padded/truncated to this width
FMIN        = 60.0
FMAX        = 8_000.0

# Train / Val / Test split ratios (patient-level)
TRAIN_RATIO = 0.60
VAL_RATIO   = 0.10
TEST_RATIO  = 0.30         # = 1 - TRAIN_RATIO - VAL_RATIO

RANDOM_SEED = 42

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════

def extract_features(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Extract a 3-channel feature tensor from a mono audio array.

    Channel 0 : Log Mel-spectrogram  (128 × MAX_FRAMES)
    Channel 1 : MFCC                 (128 × MAX_FRAMES)  — 40 coeffs, zero-padded rows
    Channel 2 : Delta-MFCC           (128 × MAX_FRAMES)

    Returns np.ndarray of shape (3, 128, MAX_FRAMES), dtype float32.
    """
    # ── Mel-spectrogram ─────────────────────────────────────────────────
    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)          # (128, T)

    # ── MFCC + delta ────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
        fmin=FMIN, fmax=FMAX,
    )                                                         # (40, T)
    delta_mfcc = librosa.feature.delta(mfcc)                 # (40, T)

    # ── Pad MFCC rows to 128 so all channels share the same shape ───────
    def pad_rows(arr: np.ndarray, target_rows: int = N_MELS) -> np.ndarray:
        rows, cols = arr.shape
        if rows < target_rows:
            pad = np.zeros((target_rows - rows, cols), dtype=np.float32)
            return np.vstack([arr, pad])
        return arr[:target_rows, :]

    mfcc_padded  = pad_rows(mfcc)
    delta_padded = pad_rows(delta_mfcc)

    # ── Normalise each channel to zero mean / unit variance ─────────────
    def standardise(arr: np.ndarray) -> np.ndarray:
        mu  = arr.mean()
        std = arr.std()
        if std < 1e-8:
            return arr - mu
        return ((arr - mu) / std).astype(np.float32)

    ch0 = standardise(log_mel)
    ch1 = standardise(mfcc_padded)
    ch2 = standardise(delta_padded)

    # ── Pad / truncate time axis to MAX_FRAMES ───────────────────────────
    def fix_time(arr: np.ndarray, max_t: int = MAX_FRAMES) -> np.ndarray:
        t = arr.shape[1]
        if t < max_t:
            pad = np.zeros((arr.shape[0], max_t - t), dtype=np.float32)
            return np.hstack([arr, pad])
        return arr[:, :max_t]

    ch0 = fix_time(ch0)
    ch1 = fix_time(ch1)
    ch2 = fix_time(ch2)

    return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)   # (3, 128, 128)


# ══════════════════════════════════════════════
# Data augmentation  (training set only)
# ══════════════════════════════════════════════

def augment_audio(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Apply random augmentation to a raw audio clip.
    All transforms are applied with 50% probability each.

    Augmentations:
      - Time stretching     ±10%
      - Pitch shifting      ±2 semitones
      - Additive Gaussian noise
    """
    rng = np.random.default_rng()

    # Time stretch
    if rng.random() < 0.5:
        rate = rng.uniform(0.9, 1.1)
        audio = librosa.effects.time_stretch(audio, rate=rate)

    # Pitch shift
    if rng.random() < 0.5:
        steps = rng.uniform(-2.0, 2.0)
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

    # Gaussian noise
    if rng.random() < 0.5:
        noise_level = rng.uniform(0.001, 0.005)
        audio = audio + noise_level * rng.standard_normal(len(audio)).astype(np.float32)

    return audio.astype(np.float32)


# ══════════════════════════════════════════════
# Patient-level train / val / test split
# ══════════════════════════════════════════════

def patient_level_split(metadata: pd.DataFrame,
                        train_ratio: float = TRAIN_RATIO,
                        val_ratio:   float = VAL_RATIO,
                        seed:        int   = RANDOM_SEED):
    """
    Split cycle indices into train / val / test ensuring no patient
    appears in more than one split (prevents data leakage).

    Returns three index arrays: train_idx, val_idx, test_idx
    """
    indices    = np.arange(len(metadata))
    patient_ids = metadata["patient_id"].values

    # First split: train+val vs test
    splitter1 = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_RATIO,
        random_state=seed,
    )
    trainval_idx, test_idx = next(
        splitter1.split(indices, groups=patient_ids)
    )

    # Second split: train vs val (from the trainval pool)
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=val_ratio_adjusted,
        random_state=seed,
    )
    rel_train, rel_val = next(
        splitter2.split(trainval_idx, groups=patient_ids[trainval_idx])
    )
    train_idx = trainval_idx[rel_train]
    val_idx   = trainval_idx[rel_val]

    return train_idx, val_idx, test_idx


# ══════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════

def build_features(input_dir: Path,
                   out_dir:   Path,
                   augment:   bool = False,
                   sr:        int  = TARGET_SR,
                   seed:      int  = RANDOM_SEED) -> None:
    """
    Full feature extraction pipeline.

    1. Load metadata.csv
    2. Extract features for every cycle
    3. Patient-level train/val/test split
    4. Optionally augment training set
    5. Save X_train/val/test.npy and y_train/val/test.npy
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ────────────────────────────────────────────────────
    meta_path = input_dir / "metadata.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"metadata.csv not found at {meta_path}\n"
            "Run src/preprocess.py first."
        )

    metadata = pd.read_csv(meta_path)
    log.info(f"Loaded metadata: {len(metadata)} cycles")

    # ── Extract raw features for all cycles ─────────────────────────────
    log.info("Extracting features from cycle WAV files...")
    all_features = []
    all_labels   = []
    valid_mask   = []                  # track which rows succeeded

    for i, row in tqdm(metadata.iterrows(), total=len(metadata),
                       desc="Extracting features", unit="cycle"):
        wav_path = input_dir / row["filename"]
        if not wav_path.exists():
            log.warning(f"  Missing file: {wav_path.name} — skipping")
            valid_mask.append(False)
            continue

        try:
            audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
            feat     = extract_features(audio, sr=sr)
            all_features.append(feat)
            all_labels.append(int(row["label"]))
            valid_mask.append(True)
        except Exception as e:
            log.warning(f"  Feature extraction failed for {wav_path.name}: {e}")
            valid_mask.append(False)

    metadata = metadata[valid_mask].reset_index(drop=True)
    X = np.stack(all_features, axis=0)          # (N, 3, 128, 128)
    y = np.array(all_labels,   dtype=np.int64)  # (N,)

    log.info(f"Feature array shape : {X.shape}")
    log.info(f"Label array shape   : {y.shape}")
    log.info(f"Label distribution  : { {i: int((y==i).sum()) for i in range(4)} }")

    # ── Patient-level split ──────────────────────────────────────────────
    log.info("Splitting into train / val / test at patient level...")
    train_idx, val_idx, test_idx = patient_level_split(metadata, seed=seed)

    log.info(f"  Train : {len(train_idx)} cycles  "
             f"({len(metadata.iloc[train_idx]['patient_id'].unique())} patients)")
    log.info(f"  Val   : {len(val_idx)} cycles  "
             f"({len(metadata.iloc[val_idx]['patient_id'].unique())} patients)")
    log.info(f"  Test  : {len(test_idx)} cycles  "
             f"({len(metadata.iloc[test_idx]['patient_id'].unique())} patients)")

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]

    # ── Data augmentation (training set only) ───────────────────────────
    if augment:
        log.info("Applying data augmentation to training set...")
        aug_features = []
        aug_labels   = []

        train_meta = metadata.iloc[train_idx].reset_index(drop=True)

        for i, row in tqdm(train_meta.iterrows(), total=len(train_meta),
                           desc="Augmenting", unit="cycle"):
            wav_path = input_dir / row["filename"]
            if not wav_path.exists():
                continue
            try:
                audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
                aug_audio = augment_audio(audio, sr=sr)
                aug_feat  = extract_features(aug_audio, sr=sr)
                aug_features.append(aug_feat)
                aug_labels.append(int(row["label"]))
            except Exception as e:
                log.warning(f"  Augmentation failed for {wav_path.name}: {e}")

        if aug_features:
            X_aug = np.stack(aug_features, axis=0)
            y_aug = np.array(aug_labels, dtype=np.int64)
            X_train = np.concatenate([X_train, X_aug], axis=0)
            y_train = np.concatenate([y_train, y_aug], axis=0)

            # Shuffle the augmented training set
            rng     = np.random.default_rng(seed)
            perm    = rng.permutation(len(X_train))
            X_train = X_train[perm]
            y_train = y_train[perm]

            log.info(f"  Training set after augmentation: {len(X_train)} cycles")

    # ── Save numpy arrays ────────────────────────────────────────────────
    log.info(f"Saving feature arrays to {out_dir} ...")

    np.save(out_dir / "X_train.npy", X_train)
    np.save(out_dir / "y_train.npy", y_train)
    np.save(out_dir / "X_val.npy",   X_val)
    np.save(out_dir / "y_val.npy",   y_val)
    np.save(out_dir / "X_test.npy",  X_test)
    np.save(out_dir / "y_test.npy",  y_test)

    # Save split metadata for reference
    metadata.iloc[train_idx].to_csv(out_dir / "meta_train.csv", index=False)
    metadata.iloc[val_idx  ].to_csv(out_dir / "meta_val.csv",   index=False)
    metadata.iloc[test_idx ].to_csv(out_dir / "meta_test.csv",  index=False)

    # ── Final summary ────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info(f"  X_train : {X_train.shape}  y_train : {y_train.shape}")
    log.info(f"  X_val   : {X_val.shape}    y_val   : {y_val.shape}")
    log.info(f"  X_test  : {X_test.shape}   y_test  : {y_test.shape}")
    log.info(f"  Tensor shape per sample: (3, {N_MELS}, {MAX_FRAMES})")
    log.info(f"  Augmentation applied   : {augment}")
    log.info("=" * 55)
    log.info("Feature extraction complete.")


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract features from preprocessed ICBHI cycle WAV files"
    )
    parser.add_argument("--input-dir",  type=Path, default=Path("data/processed"),
                        help="Directory containing cycle WAVs and metadata.csv")
    parser.add_argument("--out-dir",    type=Path, default=Path("data/features"),
                        help="Output directory for .npy feature arrays")
    parser.add_argument("--sample-rate",type=int,  default=TARGET_SR)
    parser.add_argument("--augment",    action="store_true",
                        help="Apply data augmentation to the training set")
    parser.add_argument("--seed",       type=int,  default=RANDOM_SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log.info("COPD-Detection — Feature Extraction Pipeline")
    log.info(f"  Input dir    : {args.input_dir}")
    log.info(f"  Output dir   : {args.out_dir}")
    log.info(f"  Sample rate  : {args.sample_rate} Hz")
    log.info(f"  Augmentation : {args.augment}")
    log.info(f"  Random seed  : {args.seed}")
    log.info("")

    if not args.input_dir.exists():
        log.error(f"input_dir does not exist: {args.input_dir}")
        log.error("Run src/preprocess.py first.")
        raise SystemExit(1)

    build_features(
        input_dir = args.input_dir,
        out_dir   = args.out_dir,
        augment   = args.augment,
        sr        = args.sample_rate,
        seed      = args.seed,
    )