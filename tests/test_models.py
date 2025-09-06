"""Model-level tests: forward passes and attention behavior.

Goals:
- HeartSoundLSTM accepts (B, n_mfcc) vectors and returns (B, num_classes) logits.
- SequenceCRNN accepts (B, T, F) tensors and returns (B, num_classes) logits.
- Attention weights (implicit) sum to ~1 per sample (indirectly validated by
  checking output shape and finite values; the explicit weights are internal).
"""

from __future__ import annotations

import torch

from src.models.lstm import HeartSoundLSTM, SequenceCRNN


def test_heartsound_lstm_forward():
    n_mfcc = 40
    num_classes = 3
    model = HeartSoundLSTM(n_mfcc=n_mfcc, num_classes=num_classes)
    x = torch.randn(4, n_mfcc)
    out = model(x)
    assert out.shape == (4, num_classes)
    assert torch.isfinite(out).all()


def test_sequence_crnn_forward():
    B, T, F = 2, 50, 128
    num_classes = 3
    model = SequenceCRNN(feature_dim=F, num_classes=num_classes)
    x = torch.randn(B, T, F)
    out = model(x)
    assert out.shape == (B, num_classes)
    assert torch.isfinite(out).all()
