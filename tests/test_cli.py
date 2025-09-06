"""Smoke tests for the CLI entrypoints.

We invoke the visualize command and a short training run with --epochs=1 to
ensure argument wiring functions. Uses the synthetic dataset fixture.
"""

from __future__ import annotations

import subprocess
import sys

PYTHON = sys.executable


def test_cli_visualize(tmp_dataset_root, tmp_path):
    out_dir = tmp_path / "vis_results"
    cmd = [
        PYTHON,
        "-m",
        "src.utils.cli",
        "visualize",
        "--data_dir",
        tmp_dataset_root,
        "--results_dir",
        str(out_dir),
        "--max_duration",
        "1.0",
        "--sr",
        "16000",
        "--per_class",
        "1",
        "--n_mels",
        "16",
        "--no_audio",
    ]
    subprocess.check_call(cmd)
    assert (out_dir / "visualizations" / "waveforms").exists()


def test_cli_train_short(tmp_dataset_root, tmp_path):
    out_dir = tmp_path / "train_results"
    models_dir = tmp_path / "train_models"
    cmd = [
        PYTHON,
        "-m",
        "src.utils.cli",
        "train",
        "--data_dir",
        tmp_dataset_root,
        "--results_dir",
        str(out_dir),
        "--models_dir",
        str(models_dir),
        "--max_duration",
        "1.0",
        "--sr",
        "16000",
        "--n_mfcc",
        "20",
        "--batch_size",
        "4",
        "--epochs",
        "1",
        "--lr",
        "0.001",
        "--test_size",
        "0.3",
        "--val_size",
        "0.2",
        "--seed",
        "0",
        "--feature_type",
        "mfcc",
        "--seq_mode",
        "seq_mel_mfcc",
        "--n_mels",
        "16",
        "--no_deltas",
        "--no_stats",
        "--no_class_weighting",
    ]
    subprocess.check_call(cmd)
    # best model saved under models_dir directly (save_experiment default True, so experiment folder)
    assert any(p.name == "best_model.pt" for p in (models_dir).rglob("*.pt"))
