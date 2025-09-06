"""Tests for MFCC feature extraction utilities.

Ensures:
- Single extraction returns expected dimensionality (n_mfcc,)
- Batch extraction stacks outputs correctly.
- Graceful handling of non-float input types.
"""

from __future__ import annotations

import numpy as np

from src.features.mfcc import batch_extract_mfcc, extract_mfcc

SR = 16000
N_MFCC = 13


def test_extract_mfcc_shape(small_sequence_audio):
    """extract_mfcc should return a (n_mfcc,) vector."""
    vec = extract_mfcc(small_sequence_audio, sr=SR, n_mfcc=N_MFCC)
    assert vec.shape == (N_MFCC,)
    assert np.isfinite(vec).all()


def test_extract_mfcc_int_input():
    """Int16 input should be converted to float without crashing."""
    sr = SR
    t = np.linspace(0, 0.2, int(sr * 0.2), endpoint=False)
    y_int = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    vec = extract_mfcc(y_int, sr=sr, n_mfcc=N_MFCC)
    assert vec.shape[0] == N_MFCC


def test_batch_extract_mfcc_stack(small_audio_batch):
    """batch_extract_mfcc returns (B, n_mfcc) and correct reported dim."""
    feats, dim = batch_extract_mfcc(small_audio_batch, sr=SR, n_mfcc=N_MFCC)
    assert feats.shape == (len(small_audio_batch), N_MFCC)
    assert dim == N_MFCC
    assert np.isfinite(feats).all()
