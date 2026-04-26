"""
ICBHI 2017 — unified preprocessing pipeline (Objective 1)
==========================================================
End-to-end: discovery → denoise → 5 s windows → Log-Mel + MFCCΔΔ²
→ COPD-only train augmentation → patient-level stratified split → .npy + metadata + config.

CLI:
    python src/preprocess.py --data_dir data/raw --output_dir data/features
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import subprocess
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import signal
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from tqdm import tqdm

warnings.filterwarnings("ignore")

try:
    import noisereduce as nr
except ImportError as e:  # pragma: no cover
    raise ImportError("Install noisereduce: pip install noisereduce") from e

try:
    from audiomentations import (
        AddGaussianNoise,
        Compose,
        PitchShift,
        Shift,
        TimeStretch,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError("Install audiomentations: pip install audiomentations") from e

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
TARGET_SR = 22_050
WIN_SEC = 5.0
WIN_SAMPLES = int(WIN_SEC * TARGET_SR)  # 110_250
HOP_SEC_FALLBACK = 2.5
N_FFT = 2048
HOP_LENGTH = 256
N_MELS = 128
N_MFCC = 40
N_FRAMES = 431
BANDPASS_LOW = 80.0
BANDPASS_HIGH = 2000.0
BANDPASS_ORDER = 4
MEL_FMIN = 50.0
MEL_FMAX = 2000.0


@dataclass
class PipelineConfig:
    data_dir: Path
    output_dir: Path
    seed: int = 42
    n_jobs: int = -1
    copd_aug_factor: int = 4
    no_augment: bool = False
    no_denoise: bool = False
    severity_mild_max: float = 0.25
    severity_moderate_max: float = 0.60
    limit_files: Optional[int] = None
    stationary_prop_decrease: float = 0.9
    nonstationary_prop_decrease: float = 0.85


# ═══════════════════════════════════════════════════════════════════════════
# SpecAugment (callable from training code)
# ═══════════════════════════════════════════════════════════════════════════


def spec_augment(
    mel: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    time_masks: int = 2,
    time_mask_param: int = 30,
    freq_masks: int = 2,
    freq_mask_param: int = 15,
) -> np.ndarray:
    """
    Time + frequency masking on a log-mel-like spectrogram.

    Parameters
    ----------
    mel : array, shape (1, n_mels, n_frames) or (n_mels, n_frames)
    rng : np.random.Generator, optional

    Returns
    -------
    augmented copy, float32, same shape as input (channel preserved if present).
    """
    rng = rng or np.random.default_rng()
    x = np.array(mel, dtype=np.float32, copy=True)
    if x.ndim == 3:
        m = x[0]
        squeeze = True
    else:
        m = x
        squeeze = False
    n_mels, n_frames = m.shape
    out = m.copy()
    for _ in range(max(0, time_masks)):
        w = int(rng.integers(1, min(time_mask_param, max(2, n_frames))))
        t0 = int(rng.integers(0, max(1, n_frames - w)))
        out[:, t0 : t0 + w] = 0.0
    for _ in range(max(0, freq_masks)):
        f = int(rng.integers(1, min(freq_mask_param, max(2, n_mels))))
        f0 = int(rng.integers(0, max(1, n_mels - f)))
        out[f0 : f0 + f, :] = 0.0
    if squeeze:
        return np.expand_dims(out, axis=0).astype(np.float32)
    return out.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset discovery
# ═══════════════════════════════════════════════════════════════════════════


def find_patient_diagnosis_csv(root: Path) -> Path:
    hits = sorted(root.rglob("patient_diagnosis.csv"))
    if not hits:
        raise FileNotFoundError(
            f"patient_diagnosis.csv not found under {root}. "
            "Point --data_dir at the ICBHI root or a parent folder."
        )
    return hits[0]


def discover_wavs(root: Path, limit: Optional[int] = None) -> List[Path]:
    wavs = sorted({p.resolve() for p in root.rglob("*.wav") if p.is_file()})
    if limit is not None:
        wavs = wavs[: int(limit)]
    if not wavs:
        raise FileNotFoundError(f"No .wav files found under {root}")
    return wavs


def load_diagnoses(csv_path: Path) -> Dict[str, str]:
    df = pd.read_csv(csv_path, header=None, names=["patient_id", "diagnosis"])
    df["patient_id"] = df["patient_id"].astype(str).str.strip()
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip()
    return dict(zip(df["patient_id"], df["diagnosis"]))


def parse_filename_meta(wav_name: str) -> Dict[str, str]:
    stem = Path(wav_name).stem
    parts = stem.split("_")
    return {
        "patient_id": parts[0] if parts else "unknown",
        "rec_index": parts[1] if len(parts) > 1 else "",
        "chest_loc": parts[2] if len(parts) > 2 else "",
        "acq_mode": parts[3] if len(parts) > 3 else "",
        "device": parts[4] if len(parts) > 4 else "",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Annotations & patient-level adventitious ratio
# ═══════════════════════════════════════════════════════════════════════════


def parse_annotation(txt_path: Path) -> List[Dict[str, Any]]:
    cycles: List[Dict[str, Any]] = []
    if not txt_path.exists():
        return cycles
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                cycles.append(
                    {
                        "start": float(parts[0]),
                        "end": float(parts[1]),
                        "crackle": int(float(parts[2])),
                        "wheeze": int(float(parts[3])),
                    }
                )
            except ValueError:
                continue
    return cycles


def collect_patient_cycle_stats(
    wav_paths: Sequence[Path],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Return (adventitious_cycle_count, total_cycle_count) per patient_id."""
    adv: Dict[str, int] = {}
    tot: Dict[str, int] = {}
    for wav in wav_paths:
        meta = parse_filename_meta(wav.name)
        pid = meta["patient_id"]
        txt = wav.with_suffix(".txt")
        cycles = parse_annotation(txt)
        tot.setdefault(pid, 0)
        adv.setdefault(pid, 0)
        for c in cycles:
            tot[pid] += 1
            if c["crackle"] == 1 or c["wheeze"] == 1:
                adv[pid] += 1
    return adv, tot


def severity_for_copd_patient(
    ratio: float,
    mild_max: float,
    moderate_max: float,
) -> int:
    if ratio < mild_max:
        return 1  # Mild
    if ratio <= moderate_max:
        return 2  # Moderate
    return 3  # Severe


def build_patient_severity_map(
    diagnoses: Dict[str, str],
    adv: Dict[str, int],
    tot: Dict[str, int],
    mild_max: float,
    moderate_max: float,
) -> Dict[str, int]:
    out: Dict[str, int] = {}
    all_pids = set(diagnoses.keys()) | set(tot.keys()) | set(adv.keys())
    for pid in all_pids:
        diag = diagnoses.get(pid, "")
        d = str(diag).strip().upper()
        if d != "COPD":
            out[pid] = 0
            continue
        t = tot.get(pid, 0)
        if t <= 0:
            out[pid] = 1
            continue
        r = adv.get(pid, 0) / float(t)
        out[pid] = severity_for_copd_patient(r, mild_max, moderate_max)
    return out


def binary_label(diagnosis: str) -> int:
    return 1 if str(diagnosis).strip().upper() == "COPD" else 0


# ═══════════════════════════════════════════════════════════════════════════
# Audio: denoise, segment
# ═══════════════════════════════════════════════════════════════════════════


def butter_bandpass_sos(y: np.ndarray, sr: int, low: float, high: float, order: int = 4):
    nyq = sr / 2.0
    lo = max(low / nyq, 1e-5)
    hi = min(high / nyq, 1.0 - 1e-5)
    if lo >= hi:
        raise ValueError(f"Invalid bandpass [{low},{high}] for sr={sr}")
    return signal.butter(order, [lo, hi], btype="band", output="sos")


def denoise_multistage(
    y: np.ndarray,
    sr: int,
    cfg: PipelineConfig,
) -> np.ndarray:
    if cfg.no_denoise:
        return y.astype(np.float32)
    y1 = nr.reduce_noise(
        y=y, sr=sr, stationary=True, prop_decrease=cfg.stationary_prop_decrease
    )
    y2 = nr.reduce_noise(
        y=y1,
        sr=sr,
        stationary=False,
        prop_decrease=cfg.nonstationary_prop_decrease,
    )
    sos = butter_bandpass_sos(y2, sr, BANDPASS_LOW, BANDPASS_HIGH, BANDPASS_ORDER)
    y3 = signal.sosfiltfilt(sos, y2).astype(np.float32)
    peak = float(np.max(np.abs(y3))) + 1e-12
    return (y3 / peak).astype(np.float32)


def _extract_window_centered(
    y: np.ndarray, sr: int, center_sec: float, win_samples: int
) -> Tuple[np.ndarray, float]:
    """Return audio of length win_samples (reflect-pad if needed) and window start time in seconds."""
    center = int(round(center_sec * sr))
    half = win_samples // 2
    start = center - half
    end = start + win_samples
    if start >= 0 and end <= len(y):
        seg = y[start:end].astype(np.float32)
        t0 = start / float(sr)
        return seg, t0
    seg = np.zeros(win_samples, dtype=np.float32)
    for i in range(win_samples):
        idx = start + i
        if 0 <= idx < len(y):
            seg[i] = y[idx]
        else:
            # reflect at boundaries
            j = idx
            if j < 0:
                j = -j
            if j >= len(y):
                j = 2 * (len(y) - 1) - j
            j = int(np.clip(j, 0, len(y) - 1))
            seg[i] = y[j]
    t0 = start / float(sr)
    return seg.astype(np.float32), t0


def segment_audio(
    y: np.ndarray,
    sr: int,
    cycles: List[Dict[str, Any]],
    win_samples: int = WIN_SAMPLES,
    hop_sec: float = HOP_SEC_FALLBACK,
) -> List[Dict[str, Any]]:
    """Produce fixed-length windows with cycle-aligned centers or sliding fallback."""
    out: List[Dict[str, Any]] = []
    if cycles:
        for ci, c in enumerate(cycles):
            dur = c["end"] - c["start"]
            if dur <= 0:
                continue
            center_sec = 0.5 * (c["start"] + c["end"])
            seg, t0 = _extract_window_centered(y, sr, center_sec, win_samples)
            out.append(
                {
                    "y": seg,
                    "window_start_sec": float(t0),
                    "crackle": int(c["crackle"]),
                    "wheeze": int(c["wheeze"]),
                    "cycle_index": ci,
                }
            )
        return out
    # Sliding fallback
    hop = int(hop_sec * sr)
    if len(y) < win_samples:
        pad = win_samples - len(y)
        y = np.pad(y.astype(np.float32), (0, pad), mode="reflect")
    pos = 0
    idx = 0
    while pos + win_samples <= len(y):
        seg = y[pos : pos + win_samples].astype(np.float32)
        out.append(
            {
                "y": seg,
                "window_start_sec": pos / float(sr),
                "crackle": 0,
                "wheeze": 0,
                "cycle_index": idx,
            }
        )
        pos += hop
        idx += 1
    if not out and len(y) >= win_samples:
        seg = y[:win_samples].astype(np.float32)
        out.append(
            {
                "y": seg,
                "window_start_sec": 0.0,
                "crackle": 0,
                "wheeze": 0,
                "cycle_index": 0,
            }
        )
    return out


def build_augmenter(seed: int) -> Compose:
    # audiomentations uses numpy random internally
    np.random.seed(seed)
    return Compose(
        [
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            Shift(min_shift=-0.2, max_shift=0.2, p=0.5),
        ],
        p=1.0,
    )


def augment_audio(y: np.ndarray, sr: int, augmenter: Compose) -> np.ndarray:
    return augmenter(samples=y, sample_rate=sr).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Features
# ═══════════════════════════════════════════════════════════════════════════


def fix_time_axis(spec: np.ndarray, n_frames: int = N_FRAMES) -> np.ndarray:
    t = spec.shape[1]
    if t == n_frames:
        return spec
    if t < n_frames:
        pad_w = n_frames - t
        return np.pad(spec, ((0, 0), (0, pad_w)), mode="edge")
    return spec[:, :n_frames]


def extract_log_mel(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=MEL_FMIN,
        fmax=min(MEL_FMAX, sr / 2 - 1),
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = fix_time_axis(log_mel, N_FRAMES)
    lo, hi = float(log_mel.min()), float(log_mel.max())
    if hi - lo < 1e-8:
        norm = np.zeros_like(log_mel, dtype=np.float32)
    else:
        norm = ((log_mel - lo) / (hi - lo)).astype(np.float32)
    return np.expand_dims(norm, axis=0)


def extract_mfcc_stack(y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=MEL_FMIN,
        fmax=min(MEL_FMAX, sr / 2 - 1),
    )
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)
    stack = np.stack([mfcc, d1, d2], axis=0).astype(np.float32)
    stack = np.stack([fix_time_axis(stack[i], N_FRAMES) for i in range(3)], axis=0)
    for c in range(3):
        ch = stack[c]
        mu, std = float(ch.mean()), float(ch.std())
        if std < 1e-8:
            stack[c] = (ch - mu).astype(np.float32)
        else:
            stack[c] = ((ch - mu) / std).astype(np.float32)
    return stack


# ═══════════════════════════════════════════════════════════════════════════
# Patient-level split (60/20/20)
# ═══════════════════════════════════════════════════════════════════════════


def patient_train_val_test(
    patient_ids: np.ndarray,
    y_binary: np.ndarray,
    seed: int,
) -> Tuple[set, set, set]:
    """StratifiedGroupKFold: first fold ~20% test; on remainder, first fold ~25% val → 20% of all."""
    patient_ids = np.asarray(patient_ids)
    y_binary = np.asarray(y_binary)
    X_dummy = np.zeros(len(patient_ids))
    class_counts = np.bincount(y_binary.astype(int), minlength=2)
    min_class = int(class_counts.min()) if len(class_counts) else 0

    if min_class >= 5 and len(patient_ids) >= 10:
        sgkf1 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        trainval_idx, test_idx = next(
            sgkf1.split(X_dummy, y_binary, groups=patient_ids)
        )
        p_test = set(patient_ids[test_idx].tolist())

        p_tv = patient_ids[trainval_idx]
        y_tv = y_binary[trainval_idx]
        X_tv = np.zeros(len(p_tv))
        tv_counts = np.bincount(y_tv.astype(int), minlength=2)
        if int(tv_counts.min()) >= 4 and len(p_tv) >= 8:
            sgkf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
            train_rel, val_rel = next(sgkf2.split(X_tv, y_tv, groups=p_tv))
        else:
            gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
            train_rel, val_rel = next(gss2.split(X_tv, y_tv, groups=p_tv))
        p_train = set(p_tv[train_rel].tolist())
        p_val = set(p_tv[val_rel].tolist())
    else:
        # Small-debug fallback (e.g., --limit): group split without stratification
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        trainval_idx, test_idx = next(gss1.split(X_dummy, y_binary, groups=patient_ids))
        p_test = set(patient_ids[test_idx].tolist())
        p_tv = patient_ids[trainval_idx]
        y_tv = y_binary[trainval_idx]
        X_tv = np.zeros(len(p_tv))
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed + 1)
        train_rel, val_rel = next(gss2.split(X_tv, y_tv, groups=p_tv))
        p_train = set(p_tv[train_rel].tolist())
        p_val = set(p_tv[val_rel].tolist())

    assert p_train.isdisjoint(p_val) and p_train.isdisjoint(p_test) and p_val.isdisjoint(p_test)
    return p_train, p_val, p_test


# ═══════════════════════════════════════════════════════════════════════════
# Per-file worker
# ═══════════════════════════════════════════════════════════════════════════


def process_one_wav(
    wav_path: Path,
    diagnoses: Dict[str, str],
    severity_map: Dict[str, int],
    cfg: PipelineConfig,
) -> List[Dict[str, Any]]:
    meta_file = parse_filename_meta(wav_path.name)
    pid = meta_file["patient_id"]
    diag = diagnoses.get(pid, "unknown")
    bin_lab = binary_label(diag)
    sev = int(severity_map.get(pid, 0))

    try:
        y, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    except Exception as e:
        log.warning("Failed to load %s: %s", wav_path.name, e)
        return []

    y = denoise_multistage(y.astype(np.float32), TARGET_SR, cfg)
    txt_path = wav_path.with_suffix(".txt")
    cycles = parse_annotation(txt_path)
    windows = segment_audio(y, TARGET_SR, cycles)
    rows: List[Dict[str, Any]] = []
    for w in windows:
        rows.append(
            {
                "y": w["y"],
                "patient_id": pid,
                "source_file": wav_path.name,
                "window_start_sec": w["window_start_sec"],
                "diagnosis": diag,
                "binary": bin_lab,
                "severity": sev,
                "crackle": w["crackle"],
                "wheeze": w["wheeze"],
                "cycle_index": w["cycle_index"],
            }
        )
    return rows


def git_commit_hash(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def package_versions() -> Dict[str, str]:
    out: Dict[str, str] = {}
    mapping = {
        "numpy": "numpy",
        "pandas": "pandas",
        "librosa": "librosa",
        "sklearn": "sklearn",
        "scipy": "scipy",
        "noisereduce": "noisereduce",
        "audiomentations": "audiomentations",
        "soundfile": "soundfile",
        "joblib": "joblib",
    }
    for key, modname in mapping.items():
        try:
            m = __import__(modname)
            out[key] = getattr(m, "__version__", "unknown")
        except Exception:
            out[key] = "not_installed"
    return out


def run_pipeline(cfg: PipelineConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    diag_csv = find_patient_diagnosis_csv(cfg.data_dir)
    diagnoses = load_diagnoses(diag_csv)
    wav_paths = discover_wavs(cfg.data_dir, cfg.limit_files)
    log.info("Found %d wav files under %s", len(wav_paths), cfg.data_dir)
    log.info("Diagnosis CSV: %s", diag_csv)

    adv, tot = collect_patient_cycle_stats(wav_paths)
    severity_map = build_patient_severity_map(
        diagnoses, adv, tot, cfg.severity_mild_max, cfg.severity_moderate_max
    )

    n_jobs = cfg.n_jobs if cfg.n_jobs != 0 else 1
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_one_wav)(wav, diagnoses, severity_map, cfg)
        for wav in tqdm(wav_paths, desc="Processing recordings")
    )
    flat: List[Dict[str, Any]] = [item for sub in results for item in sub]
    if not flat:
        raise RuntimeError("No windows produced — check paths and annotations.")

    base_df = pd.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k != "y"
            }
            for r in flat
        ]
    )
    audio_list = [r["y"] for r in flat]

    # Patient-level split
    patients = sorted(base_df["patient_id"].unique())
    y_pat = np.array([binary_label(diagnoses.get(p, "")) for p in patients])
    p_train, p_val, p_test = patient_train_val_test(
        np.array(patients), y_pat, cfg.seed
    )

    def split_for_pid(pid: str) -> str:
        if pid in p_train:
            return "train"
        if pid in p_val:
            return "val"
        if pid in p_test:
            return "test"
        return "unknown"

    base_df["split"] = base_df["patient_id"].map(split_for_pid)
    if (base_df["split"] == "unknown").any():
        raise RuntimeError("Some patients not assigned to a split.")

    # Expand augmentation (train + COPD only)
    augmenter = build_augmenter(cfg.seed)
    expanded_y: List[np.ndarray] = []
    expanded_meta: List[Dict[str, Any]] = []

    for i in range(len(base_df)):
        row = base_df.iloc[i]
        y = audio_list[i]
        sp = row["split"]
        is_copd = int(row["binary"]) == 1
        expanded_y.append(y)
        expanded_meta.append({**row.to_dict(), "is_augmented": False})
        if (
            not cfg.no_augment
            and sp == "train"
            and is_copd
            and cfg.copd_aug_factor > 0
        ):
            for k in range(cfg.copd_aug_factor):
                ya = augment_audio(y, TARGET_SR, augmenter)
                expanded_y.append(ya)
                expanded_meta.append(
                    {**row.to_dict(), "is_augmented": True, "aug_index": k}
                )

    meta_out = pd.DataFrame(expanded_meta)

    # Features
    X_mel_list: List[np.ndarray] = []
    X_mfcc_list: List[np.ndarray] = []
    y_bin_list: List[int] = []
    y_sev_list: List[int] = []

    for y in tqdm(expanded_y, desc="Feature extraction"):
        X_mel_list.append(extract_log_mel(y, TARGET_SR))
        X_mfcc_list.append(extract_mfcc_stack(y, TARGET_SR))

    for _, r in meta_out.iterrows():
        y_bin_list.append(int(r["binary"]))
        y_sev_list.append(int(r["severity"]))

    X_mel = np.stack(X_mel_list, axis=0).astype(np.float32)
    X_mfcc = np.stack(X_mfcc_list, axis=0).astype(np.float32)
    y_binary = np.array(y_bin_list, dtype=np.int64)
    y_severity = np.array(y_sev_list, dtype=np.int64)

    # Split arrays by meta_out['split']
    splits = {}
    for name in ("train", "val", "test"):
        mask = (meta_out["split"] == name).to_numpy()
        splits[name] = {
            "X_mel": X_mel[mask],
            "X_mfcc": X_mfcc[mask],
            "y_binary": y_binary[mask],
            "y_severity": y_severity[mask],
        }

    # Sanity checks
    for name in ("train", "val", "test"):
        m = meta_out["split"] == name
        assert splits[name]["X_mel"].shape[0] == int(m.sum())
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        pa = set(meta_out.loc[meta_out["split"] == a, "patient_id"])
        pb = set(meta_out.loc[meta_out["split"] == b, "patient_id"])
        assert pa.isdisjoint(pb), f"Patient leakage between {a} and {b}"

    # Save
    for name in ("train", "val", "test"):
        np.save(cfg.output_dir / f"X_mel_{name}.npy", splits[name]["X_mel"])
        np.save(cfg.output_dir / f"X_mfcc_{name}.npy", splits[name]["X_mfcc"])
        np.save(cfg.output_dir / f"y_binary_{name}.npy", splits[name]["y_binary"])
        np.save(cfg.output_dir / f"y_severity_{name}.npy", splits[name]["y_severity"])

    meta_path = cfg.output_dir / "metadata.csv"
    meta_out.to_csv(meta_path, index=False)

    repo_root = Path(__file__).resolve().parents[1]
    pipe_d = asdict(cfg)
    for k, v in list(pipe_d.items()):
        if isinstance(v, Path):
            pipe_d[k] = str(v.resolve())
    config_dict = {
        "pipeline": pipe_d,
        "paths": {
            "data_dir": str(cfg.data_dir.resolve()),
            "output_dir": str(cfg.output_dir.resolve()),
            "diagnosis_csv": str(diag_csv.resolve()),
        },
        "counts": {
            "n_wavs": len(wav_paths),
            "n_windows_base": len(base_df),
            "n_windows_total": len(meta_out),
            "n_patients": len(patients),
        },
        "split_patients": {
            "train": sorted(p_train),
            "val": sorted(p_val),
            "test": sorted(p_test),
        },
        "tensor_shapes": {
            "X_mel": [1, N_MELS, N_FRAMES],
            "X_mfcc": [3, N_MFCC, N_FRAMES],
        },
        "versions": package_versions(),
        "git_commit": git_commit_hash(repo_root),
    }
    with open(cfg.output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    log.info("Saved features to %s", cfg.output_dir)
    log.info("X_mel train shape: %s", splits["train"]["X_mel"].shape)
    log.info("X_mfcc train shape: %s", splits["train"]["X_mfcc"].shape)


def parse_args(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    p = argparse.ArgumentParser(description="ICBHI 2017 preprocessing → features")
    p.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    p.add_argument("--output_dir", type=Path, default=Path("data/features"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--copd_aug_factor", type=int, default=4)
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--no_denoise", action="store_true")
    p.add_argument("--severity_mild_max", type=float, default=0.25)
    p.add_argument("--severity_moderate_max", type=float, default=0.60)
    p.add_argument("--limit", type=int, default=None, help="Debug: max number of wav files")
    args = p.parse_args(argv)
    return PipelineConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        n_jobs=args.n_jobs,
        copd_aug_factor=args.copd_aug_factor,
        no_augment=args.no_augment,
        no_denoise=args.no_denoise,
        severity_mild_max=args.severity_mild_max,
        severity_moderate_max=args.severity_moderate_max,
        limit_files=args.limit,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = parse_args(argv)
    if not cfg.data_dir.exists():
        log.error("data_dir does not exist: %s", cfg.data_dir)
        sys.exit(1)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
