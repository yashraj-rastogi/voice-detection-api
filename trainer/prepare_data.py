"""
Data Engine — Validates, analyzes, and splits audio data for training.
Produces train/val/test Arrow datasets for fast I/O.
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_audio(filepath, cfg):
    """Return (valid, duration, snr) or (False, None, None) if corrupt."""
    try:
        y, sr = librosa.load(filepath, sr=cfg["data"]["sample_rate"], mono=True)
        duration = len(y) / sr

        if duration < cfg["data"]["min_duration_sec"]:
            return False, duration, None, "too_short"
        if duration > cfg["data"]["max_duration_sec"] * 3:
            # Allow up to 3x max_duration (will be truncated during training)
            pass

        # Compute Signal-to-Noise Ratio estimate
        rms = np.sqrt(np.mean(y ** 2))
        if rms < 1e-6:
            return False, duration, None, "silent"

        # Simple SNR estimate: signal power vs noise floor
        snr_db = 20.0 * np.log10(rms / 1e-6)
        return True, duration, snr_db, "ok"

    except Exception as e:
        return False, None, None, f"corrupt: {e}"


def scan_data_directory(data_dir, cfg):
    """Scan folders and build metadata DataFrame."""
    class_map = {
        "human": 0, "real": 0, "bonafide": 0,
        "ai": 1, "fake": 1, "generated": 1, "spoof": 1,
    }
    supported = set(cfg["data"]["supported_formats"])
    records = []
    rejected = []

    for folder_name in sorted(os.listdir(data_dir)):
        label_key = folder_name.lower()
        if label_key not in class_map:
            continue
        label = class_map[label_key]
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for root, _, files in os.walk(folder_path):
            for fname in sorted(files):
                ext = Path(fname).suffix.lower()
                if ext not in supported:
                    continue
                fpath = os.path.join(root, fname)
                valid, dur, snr, reason = validate_audio(fpath, cfg)

                if valid:
                    records.append({
                        "file": os.path.abspath(fpath),
                        "label": label,
                        "label_name": cfg["data"]["class_labels"][label],
                        "duration_sec": round(dur, 2),
                        "snr_db": round(snr, 1) if snr else None,
                        "language": "unknown",  # User can tag manually
                    })
                else:
                    rejected.append({"file": fpath, "reason": reason})

    return pd.DataFrame(records), rejected


def print_report(df, rejected):
    """Print a human-readable data quality report."""
    print("\n" + "=" * 60)
    print("  📊  DATA QUALITY REPORT")
    print("=" * 60)

    if df.empty:
        print("  ❌  No valid audio files found!")
        return

    counts = Counter(df["label_name"])
    total = len(df)
    print(f"\n  Total valid samples : {total}")
    for cls, cnt in counts.items():
        pct = cnt / total * 100
        print(f"    {cls:15s} : {cnt:5d}  ({pct:.1f}%)")

    # Balance warning
    vals = list(counts.values())
    if len(vals) >= 2 and max(vals) / max(min(vals), 1) > 2:
        print("\n  ⚠️  WARNING: Class imbalance exceeds 2:1 ratio!")
        print("     Consider collecting more samples for the minority class.")

    # Duration stats
    print(f"\n  Duration (sec)  min={df['duration_sec'].min():.1f}  "
          f"mean={df['duration_sec'].mean():.1f}  "
          f"max={df['duration_sec'].max():.1f}")

    if rejected:
        print(f"\n  ⛔  Rejected files : {len(rejected)}")
        for r in rejected[:5]:
            print(f"     {r['file']}  →  {r['reason']}")
        if len(rejected) > 5:
            print(f"     ... and {len(rejected) - 5} more")

    print("=" * 60 + "\n")


def split_data(df, cfg):
    """Stratified train/val/test split."""
    ratios = cfg["data"]["split_ratios"]  # [0.70, 0.15, 0.15]
    seed = cfg["data"]["seed"]

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df, test_size=ratios[1] + ratios[2],
        stratify=df["label"], random_state=seed
    )
    # Second split: val vs test
    relative_test = ratios[2] / (ratios[1] + ratios[2])
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test,
        stratify=temp_df["label"], random_state=seed
    )
    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Data Engine: Validate, analyze, and split audio data")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data dir from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["paths"]["data_dir"]

    if not os.path.exists(data_dir):
        os.makedirs(os.path.join(data_dir, "human"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "ai"), exist_ok=True)
        print("📂 Created data directory structure. Add your audio files and run again.")
        return

    print("🔍 Scanning and validating audio files...")
    df, rejected = scan_data_directory(data_dir, cfg)
    print_report(df, rejected)

    if len(df) < 4:
        print("❌ Need at least 4 valid samples (2 per class) to create splits.")
        return

    # Save full metadata
    metadata_path = os.path.join(cfg["paths"]["output_dir"], "metadata")
    os.makedirs(metadata_path, exist_ok=True)

    df.to_csv(os.path.join(metadata_path, "all_data.csv"), index=False)

    # Split
    train_df, val_df, test_df = split_data(df, cfg)
    train_df.to_csv(os.path.join(metadata_path, "train.csv"), index=False)
    val_df.to_csv(os.path.join(metadata_path, "val.csv"), index=False)
    test_df.to_csv(os.path.join(metadata_path, "test.csv"), index=False)

    print(f"✅ Splits saved to {metadata_path}/")
    print(f"   Train : {len(train_df)}")
    print(f"   Val   : {len(val_df)}")
    print(f"   Test  : {len(test_df)}")

    # Save rejected list
    if rejected:
        with open(os.path.join(metadata_path, "rejected.json"), "w") as f:
            json.dump(rejected, f, indent=2)


if __name__ == "__main__":
    main()
