"""Tests for data loading and dataset building utilities.

Focus:
- Ensures that building the dataset returns parallel X (object array) and y (int labels).
- Verifies padding / length normalization logic when max_duration is enforced.
- Confirms CATEGORY_MAP consistency.
"""

from __future__ import annotations

import numpy as np

from src.data import dataset as ds

MAX_DUR = 1.0  # keep tiny for speed
SR = 16000


def test_category_map_labels_are_consecutive():
    """CATEGORY_MAP values should form a 0..N-1 sequence for stable indexing."""
    vals = sorted(ds.CATEGORY_MAP.values())
    assert vals == list(range(len(vals)))


def test_build_dataset_shapes(tmp_dataset_root):
    """build_dataset should return object array X and numeric y with same length."""
    X, y = ds.build_dataset(tmp_dataset_root, max_duration=MAX_DUR, sr=SR)
    assert isinstance(X, np.ndarray) and X.dtype == object
    assert isinstance(y, np.ndarray)
    assert len(X) == len(y) > 0


def test_build_dataset_padding_length(tmp_dataset_root):
    """All returned audio clips must be (approximately) max_duration * sr samples.

    We allow ±1 sample tolerance due to potential resampling rounding.
    """
    X, _ = ds.build_dataset(tmp_dataset_root, max_duration=MAX_DUR, sr=SR)
    target = int(SR * MAX_DUR)
    for a in X:
        assert abs(len(a) - target) <= 1
