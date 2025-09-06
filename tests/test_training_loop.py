"""Lightweight tests for the training utilities.

We do NOT execute full multi-epoch training on the real dataset (too slow).
Instead we: (1) build the tiny synthetic dataset fixture, (2) run a drastically
shortened training (1 epoch) for both MFCC and sequence modes, and (3) verify
artifacts & metrics dictionary structure.
"""

from __future__ import annotations

import os

from src.training.train import run_training


def _run_short_training(data_dir: str, feature_type: str):
    return run_training(
        data_dir=data_dir,
        results_dir="test_results",
        models_dir="test_models",
        max_duration=1.0,
        sr=16000,
        n_mfcc=20,
        batch_size=4,
        epochs=1,
        lr=1e-3,
        test_size=0.3,
        val_size=0.2,
        seed=0,
        feature_type=feature_type,
        seq_mode="seq_mel_mfcc",
        n_mels=16,
        add_deltas=False,
        add_stats=False,
        class_weighting=False,
        focal_loss=False,
        save_experiment=False,
        experiment_label=f"unit_{feature_type}",
    )


def test_short_training_mfcc(tmp_dataset_root):
    res = _run_short_training(tmp_dataset_root, "mfcc")
    assert "test_acc" in res and "test_loss" in res
    assert isinstance(res["test_acc"], float)
    assert os.path.exists(os.path.join(res["models_dir"], "best_model.pt"))


def test_short_training_sequence(tmp_dataset_root):
    res = _run_short_training(tmp_dataset_root, "sequence")
    assert "history" in res and "report" in res
    assert os.path.exists(os.path.join(res["models_dir"], "best_model.pt"))
