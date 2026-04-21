"""
preprocess.py — ICBHI 2017 Respiratory Sound Database Preprocessing Pipeline
=============================================================================
Steps performed:
  1. Read all WAV files + annotation (.txt) files from data/raw/
  2. Parse annotation timestamps → individual breathing cycle segments
  3. Resample each cycle to 22,050 Hz
  4. Apply bandpass filter (100–8,000 Hz)
  5. Apply spectral subtraction noise reduction
  6. Normalise peak amplitude to [-1.0, 1.0]
  7. Discard cycles shorter than min_duration seconds
  8. Save each cycle as a WAV file in data/processed/
  9. Save a metadata CSV mapping cycle files → labels + patient info

Usage (PowerShell):
    python src/preprocess.py
    python src/preprocess.py --raw-dir data/raw --out-dir data/processed --min-duration 0.5

Author: VCET COPD-Detection Team
"""

import os
import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TARGET_SR = 22_050          # target sample rate (Hz)
LOWCUT    = 100             # bandpass low cutoff (Hz)
HIGHCUT   = 8_000           # bandpass high cutoff (Hz)
FILTER_ORDER = 4            # Butterworth filter order

# ICBHI chest location codes → human-readable
LOCATION_MAP = {
    "Al": "Anterior Left",
    "Ar": "Anterior Right",
    "Pl": "Posterior Left",
    "Pr": "Posterior Right",
    "Ll": "Lateral Left",
    "Lr": "Lateral Right",
    "Tc": "Trachea",
}

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Audio processing utilities
# ══════════════════════════════════════════════

def bandpass_filter(signal: np.ndarray, sr: int,
                    lowcut: float = LOWCUT,
                    highcut: float = HIGHCUT,
                    order: int = FILTER_ORDER) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to a 1-D audio signal.
    Uses second-order sections (sosfilt) for numerical stability.
    """
    nyq = sr / 2.0
    low  = lowcut  / nyq
    high = highcut / nyq
    # Clamp to valid range to avoid edge-case errors
    low  = max(low,  1e-4)
    high = min(high, 1.0 - 1e-4)
    sos = butter(order, [low, high], btype="bandpass", output="sos")
    return sosfilt(sos, signal).astype(np.float32)


def spectral_subtraction(signal: np.ndarray, sr: int,
                         noise_duration: float = 0.1) -> np.ndarray:
    """
    Simple spectral subtraction noise reduction.
    Estimates noise PSD from the first `noise_duration` seconds,
    then subtracts it from the full signal spectrum.
    """
    n_fft    = 1024
    hop_len  = 256
    noise_samples = int(noise_duration * sr)

    # Estimate noise from the first segment
    noise_segment = signal[:noise_samples] if len(signal) > noise_samples else signal
    noise_stft    = librosa.stft(noise_segment, n_fft=n_fft, hop_length=hop_len)
    noise_power   = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)

    # Full signal STFT
    signal_stft  = librosa.stft(signal, n_fft=n_fft, hop_length=hop_len)
    signal_power = np.abs(signal_stft) ** 2
    phase        = np.angle(signal_stft)

    # Subtract noise power (floor at 0)
    clean_power  = np.maximum(signal_power - noise_power, 0.0)
    clean_mag    = np.sqrt(clean_power)
    clean_stft   = clean_mag * np.exp(1j * phase)

    reconstructed = librosa.istft(clean_stft, hop_length=hop_len,
                                  length=len(signal))
    return reconstructed.astype(np.float32)


def normalise_amplitude(signal: np.ndarray) -> np.ndarray:
    """
    Peak-normalise signal to the range [-1.0, 1.0].
    Returns the original signal unchanged if it is silent.
    """
    peak = np.max(np.abs(signal))
    if peak < 1e-8:
        return signal
    return (signal / peak).astype(np.float32)


def load_and_preprocess(wav_path: Path, sr: int = TARGET_SR,
                        denoise: bool = True) -> np.ndarray:
    """
    Load a WAV file, resample to `sr`, apply bandpass filter,
    optional spectral subtraction, and peak normalisation.
    Returns a float32 mono array.
    """
    audio, orig_sr = librosa.load(str(wav_path), sr=None, mono=True)

    # Resample if needed
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)

    audio = bandpass_filter(audio, sr)

    if denoise:
        audio = spectral_subtraction(audio, sr)

    audio = normalise_amplitude(audio)
    return audio


# ══════════════════════════════════════════════
# Annotation parsing
# ══════════════════════════════════════════════

def parse_annotation(txt_path: Path) -> list[dict]:
    """
    Parse an ICBHI annotation file.

    File format (tab-separated, no header):
        <start_sec>  <end_sec>  <crackle>  <wheeze>

    Returns a list of dicts, one per breathing cycle.
    """
    cycles = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                cycles.append({
                    "start":   float(parts[0]),
                    "end":     float(parts[1]),
                    "crackle": int(parts[2]),
                    "wheeze":  int(parts[3]),
                })
            except ValueError:
                continue
    return cycles


def parse_filename(wav_name: str) -> dict:
    """
    Extract metadata encoded in the ICBHI filename convention:
        {patient_id}_{rec_index}_{chest_loc}_{acq_mode}_{device}.wav

    Example: 101_1b1_Al_sc_Meditron.wav
    """
    stem  = Path(wav_name).stem
    parts = stem.split("_")
    result = {
        "patient_id":  parts[0] if len(parts) > 0 else "unknown",
        "rec_index":   parts[1] if len(parts) > 1 else "unknown",
        "chest_loc":   LOCATION_MAP.get(parts[2], parts[2]) if len(parts) > 2 else "unknown",
        "acq_mode":    parts[3] if len(parts) > 3 else "unknown",
        "device":      parts[4] if len(parts) > 4 else "unknown",
    }
    return result


def assign_label(crackle: int, wheeze: int) -> int:
    """
    Map (crackle, wheeze) flags to a 4-class label.

        0 → Normal        (no crackle, no wheeze)
        1 → Crackle only
        2 → Wheeze only
        3 → Both

    This is the standard ICBHI 4-class scheme used as a
    proxy for COPD severity in the absence of GOLD staging.
    """
    if crackle == 0 and wheeze == 0:
        return 0
    elif crackle == 1 and wheeze == 0:
        return 1
    elif crackle == 0 and wheeze == 1:
        return 2
    else:
        return 3


LABEL_NAMES = {0: "Normal", 1: "Crackle", 2: "Wheeze", 3: "Both"}


# ══════════════════════════════════════════════
# Patient diagnosis loading
# ══════════════════════════════════════════════

def load_patient_diagnoses(raw_dir: Path) -> dict:
    """
    Load patient_diagnosis.csv (or .txt) from raw_dir.
    Returns a dict mapping patient_id (str) → diagnosis (str).
    Falls back gracefully if the file is missing.
    """
    for name in ["patient_diagnosis.csv", "patient_diagnosis.txt"]:
        p = raw_dir / name
        if p.exists():
            try:
                df = pd.read_csv(p, header=None, names=["patient_id", "diagnosis"])
                df["patient_id"] = df["patient_id"].astype(str).str.strip()
                df["diagnosis"]  = df["diagnosis"].astype(str).str.strip()
                log.info(f"Loaded patient diagnoses from {p.name}  ({len(df)} patients)")
                return dict(zip(df["patient_id"], df["diagnosis"]))
            except Exception as e:
                log.warning(f"Could not parse {p.name}: {e}")
    log.warning("patient_diagnosis file not found — diagnosis column will be 'unknown'")
    return {}


# ══════════════════════════════════════════════
# Main pipeline
# ══════════════════════════════════════════════

def process_dataset(raw_dir: Path,
                    out_dir: Path,
                    min_duration: float = 0.5,
                    sr: int = TARGET_SR,
                    denoise: bool = True) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline over all WAV files in raw_dir.

    Parameters
    ----------
    raw_dir      : path containing .wav + .txt annotation files
    out_dir      : where to save cycle WAV files
    min_duration : minimum cycle length in seconds (shorter cycles are dropped)
    sr           : target sample rate
    denoise      : whether to apply spectral subtraction

    Returns
    -------
    metadata DataFrame saved to out_dir/metadata.csv
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    diagnoses = load_patient_diagnoses(raw_dir)

    # Collect all WAV files
    wav_files = sorted(raw_dir.glob("*.wav"))
    if not wav_files:
        log.error(f"No WAV files found in {raw_dir}")
        log.error("Make sure you extracted ICBHI_final_database.zip into data/raw/")
        raise FileNotFoundError(f"No WAV files in {raw_dir}")

    log.info(f"Found {len(wav_files)} WAV files in {raw_dir}")

    records     = []
    total_saved = 0
    total_skip  = 0

    for wav_path in tqdm(wav_files, desc="Processing recordings", unit="file"):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            log.warning(f"  No annotation file for {wav_path.name} — skipping")
            continue

        # Parse filename metadata
        meta = parse_filename(wav_path.name)
        pid  = meta["patient_id"]
        diag = diagnoses.get(pid, "unknown")

        # Load and preprocess the full recording
        try:
            full_audio = load_and_preprocess(wav_path, sr=sr, denoise=denoise)
        except Exception as e:
            log.warning(f"  Failed to load {wav_path.name}: {e}")
            continue

        # Parse annotation → extract individual cycles
        cycles = parse_annotation(txt_path)
        if not cycles:
            log.warning(f"  No valid cycles in {txt_path.name}")
            continue

        for idx, cycle in enumerate(cycles):
            start_sec = cycle["start"]
            end_sec   = cycle["end"]
            duration  = end_sec - start_sec

            # Drop cycles that are too short
            if duration < min_duration:
                total_skip += 1
                continue

            # Convert time → samples
            start_sample = int(start_sec * sr)
            end_sample   = int(end_sec   * sr)

            # Guard against annotation overshoot
            end_sample = min(end_sample, len(full_audio))
            if start_sample >= end_sample:
                total_skip += 1
                continue

            cycle_audio = full_audio[start_sample:end_sample]

            # Build output filename
            stem     = wav_path.stem
            out_name = f"{stem}_cycle_{idx:03d}.wav"
            out_path = out_dir / out_name

            # Save cycle WAV
            sf.write(str(out_path), cycle_audio, sr, subtype="PCM_16")

            label = assign_label(cycle["crackle"], cycle["wheeze"])

            records.append({
                "filename":    out_name,
                "patient_id":  pid,
                "diagnosis":   diag,
                "chest_loc":   meta["chest_loc"],
                "device":      meta["device"],
                "acq_mode":    meta["acq_mode"],
                "start_sec":   round(start_sec, 4),
                "end_sec":     round(end_sec,   4),
                "duration":    round(duration,  4),
                "crackle":     cycle["crackle"],
                "wheeze":      cycle["wheeze"],
                "label":       label,
                "label_name":  LABEL_NAMES[label],
            })
            total_saved += 1

    if not records:
        log.error("No cycles were saved. Check your raw_dir path and annotation files.")
        raise RuntimeError("Preprocessing produced no output.")

    metadata = pd.DataFrame(records)
    csv_path = out_dir / "metadata.csv"
    metadata.to_csv(csv_path, index=False)

    # ── Summary ─────────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info(f"  Cycles saved   : {total_saved}")
    log.info(f"  Cycles skipped : {total_skip}  (too short < {min_duration}s)")
    log.info(f"  Output dir     : {out_dir}")
    log.info(f"  Metadata CSV   : {csv_path}")
    log.info("")
    log.info("  Label distribution:")
    for label_id, name in LABEL_NAMES.items():
        count = (metadata["label"] == label_id).sum()
        pct   = count / len(metadata) * 100
        log.info(f"    {label_id} — {name:<10}: {count:>5} cycles  ({pct:.1f}%)")
    log.info("=" * 55)

    return metadata


# ══════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess ICBHI 2017 respiratory sound dataset"
    )
    parser.add_argument("--raw-dir",      type=Path, default=Path("data/raw"),
                        help="Directory containing raw ICBHI WAV + TXT files")
    parser.add_argument("--out-dir",      type=Path, default=Path("data/processed"),
                        help="Output directory for cycle WAV files and metadata.csv")
    parser.add_argument("--sample-rate",  type=int,  default=TARGET_SR,
                        help="Target sample rate in Hz (default: 22050)")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Minimum cycle duration in seconds (default: 0.5)")
    parser.add_argument("--no-denoise",   action="store_true",
                        help="Skip spectral subtraction noise reduction")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    log.info("COPD-Detection — Preprocessing Pipeline")
    log.info(f"  Raw dir      : {args.raw_dir}")
    log.info(f"  Output dir   : {args.out_dir}")
    log.info(f"  Sample rate  : {args.sample_rate} Hz")
    log.info(f"  Min duration : {args.min_duration} s")
    log.info(f"  Denoise      : {not args.no_denoise}")
    log.info("")

    if not args.raw_dir.exists():
        log.error(f"raw_dir does not exist: {args.raw_dir}")
        log.error("Download the ICBHI 2017 dataset and extract it into data/raw/")
        raise SystemExit(1)

    metadata = process_dataset(
        raw_dir      = args.raw_dir,
        out_dir      = args.out_dir,
        min_duration = args.min_duration,
        sr           = args.sample_rate,
        denoise      = not args.no_denoise,
    )

    log.info("Preprocessing complete.")