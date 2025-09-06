import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def extract_mfcc(y: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    if not np.issubdtype(y.dtype, np.floating):
        # convert if still not float32
        y = y.astype(np.float32)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # mean over time axis -> shape (n_mfcc,)
    return mfccs.mean(axis=1)


def batch_extract_mfcc(batch_audio, sr: int, n_mfcc: int = 40) -> tuple[np.ndarray, int]:
    features = []
    for i, y in enumerate(batch_audio):
        try:
            features.append(extract_mfcc(y, sr=sr, n_mfcc=n_mfcc))
        except Exception as e:
            logger.warning("MFCC extraction failed for index %d: %s", i, e)
            # append zeros to keep alignment
            features.append(np.zeros(n_mfcc, dtype=np.float32))
    return np.stack(features), n_mfcc
