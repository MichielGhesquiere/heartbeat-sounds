"""Additional coverage tests targeting previously untested branches.

These tests focus on:
 - Focal loss & class weighting creation
 - SequenceCRNN projection branch (proj_dim)
 - MFCC batch extraction error fallback (zero row)
 - Sequential feature extraction fallback (forced exception)
 - run_training with save_experiment=True producing artifacts & CSV row
 - plot_experiment_recalls after two experiments
 - visualize_examples with save_audio toggled off
 - logging setup idempotency
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.features import sequential as seq_mod
from src.features.mfcc import batch_extract_mfcc
from src.models.lstm import SequenceCRNN
from src.training.train import plot_experiment_recalls, run_training
from src.utils.logging_config import setup_logging
from src.utils.visualize import visualize_examples


def test_sequence_crnn_projection_branch():
    model = SequenceCRNN(feature_dim=32, num_classes=3, proj_dim=16)
    x = torch.randn(2, 10, 32)
    out = model(x)
    assert out.shape == (2, 3)


def test_mfcc_batch_error_fallback(monkeypatch):
    # Force extract_mfcc to raise for one element
    import src.features.mfcc as mf

    def boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(mf, "extract_mfcc", boom)
    feats, dim = batch_extract_mfcc([np.zeros(1600)], sr=16000, n_mfcc=5)
    assert feats.shape == (1, 5)
    assert np.allclose(feats[0], 0.0)


def test_sequence_feature_fallback(monkeypatch):
    # Force failure inside extract_sequence_features by patching melspectrogram
    def raise_err(*a, **k):
        raise ValueError("intentional")

    monkeypatch.setattr(seq_mod.librosa.feature, "melspectrogram", raise_err)
    y = np.zeros(1600, dtype=np.float32)
    m = seq_mod.extract_sequence_features(y, sr=16000)
    assert m.shape[1] == 1  # fallback produces (T,1)


def test_run_training_artifacts_and_csv(tmp_dataset_root, tmp_path):
    # First experiment (result unused; run to create initial CSV row)
    run_training(
        data_dir=tmp_dataset_root,
        results_dir=str(tmp_path / "res"),
        models_dir=str(tmp_path / "models"),
        max_duration=1.0,
        sr=16000,
        n_mfcc=10,
        batch_size=4,
        epochs=1,
        feature_type="mfcc",
        class_weighting=True,
        focal_loss=True,
        gamma=1.5,
        save_experiment=True,
        experiment_label="expA",
    )
    # Second experiment to append CSV
    res2 = run_training(
        data_dir=tmp_dataset_root,
        results_dir=str(tmp_path / "res"),
        models_dir=str(tmp_path / "models"),
        max_duration=1.0,
        sr=16000,
        n_mfcc=10,
        batch_size=4,
        epochs=1,
        feature_type="sequence",
        class_weighting=True,
        focal_loss=False,
        save_experiment=True,
        experiment_label="expB",
        add_deltas=False,
        add_stats=False,
        n_mels=16,
    )
    csv_path = tmp_path / "res" / "experiment_recalls_detailed.csv"
    assert csv_path.exists()
    lines = csv_path.read_text().strip().splitlines()
    assert len(lines) >= 3  # header + at least 2 experiments
    # plot experiments
    plot_experiment_recalls(results_dir=str(tmp_path / "res"))
    assert (tmp_path / "res" / "recall_progress.png").exists()

    # Check confusion matrix image from second experiment
    assert any(Path(res2["results_dir"]).glob("confusion_matrix.png"))


def test_visualize_examples_no_audio(tmp_dataset_root, tmp_path):
    visualize_examples(
        data_dir=tmp_dataset_root,
        results_dir=str(tmp_path / "viz"),
        max_duration=1.0,
        sr=16000,
        per_class=1,
        n_mels=16,
        save_audio=False,
    )
    wave_dir = tmp_path / "viz" / "visualizations" / "waveforms"
    assert wave_dir.exists() and any(wave_dir.glob("*.png"))


def test_setup_logging_idempotent():
    setup_logging()
    before = len(logging.getLogger().handlers)
    setup_logging()
    after = len(logging.getLogger().handlers)
    assert before == after  # no duplicate handlers
