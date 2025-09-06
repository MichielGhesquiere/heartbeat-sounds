"""Pytest configuration and shared fixtures for the heartbeat-sounds project.

This file provides small synthetic audio clips and a temporary dataset layout so
that unit tests can run quickly without depending on the large original corpus.
The goal is to validate logic (shapes, padding, feature extraction, model
forward passes, training loop invariants) rather than achieve real accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# Ensure project root (parent of tests/) is on sys.path so `import src.*` works
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SR = 16000


@pytest.fixture(scope="session")
def tmp_dataset_root(tmp_path_factory: pytest.TempPathFactory):
    """Create a miniature dataset folder structure with a few short wav files.

    Structure mimics the expected on-disk layout:
        <root>/set_a/<category>__*.wav
        <root>/set_b/<category>__*.wav

    We generate 0.6s sine waves with slight frequency differences per class so
    librosa loads them consistently. Each file is padded by dataset code to the
    configured max duration during tests.
    """
    root = tmp_path_factory.mktemp("mini_data")
    for subset in ("set_a", "set_b"):
        (root / subset).mkdir(exist_ok=True)
    durations = {"artifact": 0.4, "murmur": 0.5, "normal": 0.6}
    freqs = {"artifact": 220, "murmur": 330, "normal": 440}
    # Create two files per class across subsets to enable stratified splits in normal runs.
    for subset in ("set_a", "set_b"):
        for label, dur in durations.items():
            for rep in range(1):  # adjustable reps if needed
                t = np.linspace(0, dur, int(SR * dur), endpoint=False)
                # Slightly modulate frequency to avoid identical arrays
                freq = freqs[label] + rep * 5
                y = 0.2 * np.sin(2 * np.pi * freq * t).astype(np.float32)
                out = root / subset / f"{label}__synthetic_{subset}_{rep}.wav"
                sf.write(out, y, SR)
    yield str(root)
    # cleanup handled by tmp_path_factory


@pytest.fixture
def small_audio_batch():
    """Return a list of 3 synthetic audio numpy arrays for feature tests."""
    rng = np.random.default_rng(0)
    return [rng.standard_normal(16000).astype(np.float32) * 0.01 for _ in range(3)]


@pytest.fixture
def small_sequence_audio():
    """Return a short random waveform for sequence feature extraction tests."""
    rng = np.random.default_rng(1)
    return (rng.standard_normal(16000) * 0.01).astype(np.float32)
