"""Utility to collect a small, version-controlled snapshot of key result images
for inclusion in the README without committing the full results directory.

Usage (after running training & visualize):

  python -m src.utils.export_readme_assets \
      --results_dir results \
      --out_dir docs \
      --waveform_count 1

It will copy (if found):
  - recall_progress.png
  - recall_gains.png (optional)
  - One waveform + mel spectrogram per class (artifact/murmur/normal) matching
    patterns waveform_<label>_0.png and mel_<label>_0.png (or first available).

Creates a manifest file docs/README_ASSETS_MANIFEST.txt describing the sources.
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from datetime import datetime

CORE_PROGRESS_FILES = ["recall_progress.png", "recall_gains.png", "confusion_matrix.png"]
WAVEFORM_PREFIX = "waveform_"
MEL_PREFIX = "mel_"

DEFAULT_LABELS = ["artifact", "murmur", "normal"]


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _copy_if_exists(src: str, dst: str, copied: list[str]):
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def _find_first(patterns):
    for pat in patterns:
        matches = glob.glob(pat)
        if matches:
            return matches[0]
    return None


def export_assets(
    results_dir: str, out_dir: str, waveform_count: int = 1, include_spectrograms: bool = True
):
    out_img_dir = os.path.join(out_dir)
    _ensure_dir(out_img_dir)
    copied: list[str] = []

    # Copy core progress images
    for fname in CORE_PROGRESS_FILES:
        src = os.path.join(results_dir, fname)
        dst = os.path.join(out_img_dir, fname)
        _copy_if_exists(src, dst, copied)

    # Visualization subfolders (expected under results/visualizations/...)
    wave_dir = os.path.join(results_dir, "visualizations", "waveforms")
    mel_dir = os.path.join(results_dir, "visualizations", "spectrograms")

    for label in DEFAULT_LABELS:
        # Prefer numbered 0..N
        for idx in range(waveform_count):
            # Waveform
            candidates = [
                os.path.join(wave_dir, f"{WAVEFORM_PREFIX}{label}_{idx}.png"),
                os.path.join(wave_dir, f"{WAVEFORM_PREFIX}{label}.png"),
            ]
            wave_src = next((c for c in candidates if os.path.exists(c)), None)
            if wave_src:
                wave_dst = os.path.join(out_img_dir, f"waveform_{label}_{idx}.png")
                shutil.copyfile(wave_src, wave_dst)
                copied.append(wave_dst)
            # Mel spec
            if include_spectrograms:
                spec_candidates = [
                    os.path.join(mel_dir, f"{MEL_PREFIX}{label}_{idx}.png"),
                    os.path.join(mel_dir, f"{MEL_PREFIX}{label}.png"),
                ]
                spec_src = next((c for c in spec_candidates if os.path.exists(c)), None)
                if spec_src:
                    spec_dst = os.path.join(out_img_dir, f"mel_{label}_{idx}.png")
                    shutil.copyfile(spec_src, spec_dst)
                    copied.append(spec_dst)

    # Manifest
    manifest_path = os.path.join(out_img_dir, "README_ASSETS_MANIFEST.txt")
    with open(manifest_path, "w") as mf:
        mf.write(f"Generated: {datetime.utcnow().isoformat()}Z\n")
        mf.write(f"Source results dir: {os.path.abspath(results_dir)}\n")
        mf.write("Copied files (relative to repo root):\n")
        for f in copied:
            mf.write(f"  {os.path.relpath(f)}\n")
    return copied


def main():
    parser = argparse.ArgumentParser(description="Export compact set of result images for README.")
    parser.add_argument("--results_dir", type=str, default="results", help="Root results directory")
    parser.add_argument(
        "--out_dir", type=str, default="docs", help="Destination directory (tracked)"
    )
    parser.add_argument("--waveform_count", type=int, default=1, help="Waveforms per class")
    parser.add_argument("--no_spectrograms", action="store_true", help="Exclude mel spectrograms")
    args = parser.parse_args()
    copied = export_assets(
        results_dir=args.results_dir,
        out_dir=args.out_dir,
        waveform_count=args.waveform_count,
        include_spectrograms=not args.no_spectrograms,
    )
    print("Copied:")
    for c in copied:
        print(" -", c)


if __name__ == "__main__":
    main()
