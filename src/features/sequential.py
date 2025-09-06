import numpy as np
import librosa
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def extract_sequence_features(
    y: np.ndarray,
    sr: int,
    mode: str = 'seq_mel_mfcc',
    n_mels: int = 64,
    n_mfcc: int = 20,
    add_deltas: bool = True,
    add_spectral_stats: bool = True,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> np.ndarray:
    """Return time-major feature matrix (T, F).

    Modes:
      seq_mel: log-mel base
      seq_mfcc: mfcc base
      seq_mel_mfcc: concatenated log-mel + mfcc
    Additional (optional): deltas (1st & 2nd order) and spectral stats per frame.
    """
    try:
        if not np.issubdtype(y.dtype, np.floating):
            y = y.astype(np.float32)
        else:
            y = y.astype(np.float32, copy=False)

        # Base time-frequency representations
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length, power=2.0
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        feats: List[np.ndarray] = []
        if mode in ('seq_mel', 'seq_mel_mfcc'):
            feats.append(log_mel)
        if mode in ('seq_mfcc', 'seq_mel_mfcc'):
            # MFCC from log-mel (stability) – librosa expects log power or power; supply log_mel via S=.
            mfcc = librosa.feature.mfcc(S=log_mel, sr=sr, n_mfcc=n_mfcc)
            feats.append(mfcc)

        if add_deltas:
            new_feats = []
            for f in feats:
                delta1 = librosa.feature.delta(f)
                delta2 = librosa.feature.delta(f, order=2)
                new_feats.extend([f, delta1, delta2])
            feats = new_feats

        if add_spectral_stats:
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length)
            rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)
            # Contrast can have a slightly different frame alignment; compute and trim/pad.
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
            stats = [spectral_centroid, spectral_bandwidth, spectral_rolloff, zcr, rms, contrast]
            # Align time frames by min length
            min_frames = min(s.shape[1] for s in stats + feats)
            aligned_stats = [s[:, :min_frames] for s in stats]
            feats = [f[:, :min_frames] for f in feats]  # align bases
            feats.extend(aligned_stats)

        # Concatenate along feature (rows) -> shape (F, T)
        feat_matrix = np.concatenate(feats, axis=0)
        # Transpose to (T, F)
        return feat_matrix.T.astype(np.float32)
    except Exception as e:
        logger.warning("Sequence feature extraction failed: %s", e)
        # Fallback minimal feature: mean amplitude per 512 sample window
        if y.size == 0:
            return np.zeros((1, 1), dtype=np.float32)
        frames = int(np.ceil(len(y) / hop_length))
        reshaped = np.pad(y, (0, frames * hop_length - len(y)))
        reshaped = reshaped.reshape(frames, hop_length)
        mean_abs = np.mean(np.abs(reshaped), axis=1, keepdims=True)
        return mean_abs.astype(np.float32)


def batch_extract_sequence(
    batch_audio: List[np.ndarray],
    sr: int,
    mode: str,
    n_mels: int,
    n_mfcc: int,
    add_deltas: bool,
    add_spectral_stats: bool,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> Tuple[np.ndarray, int, int]:
    """Extract sequence features for a list of audio arrays.

    Returns
    -------
    features: np.ndarray shape (N, T, F)
    T: int number of frames
    F: int feature dimension
    """
    feature_list: List[np.ndarray] = []
    max_T: int = 0
    feature_dim: int = 0
    for i, y in enumerate(batch_audio):
        try:
            m = extract_sequence_features(
                y,
                sr=sr,
                mode=mode,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
                add_deltas=add_deltas,
                add_spectral_stats=add_spectral_stats,
                n_fft=n_fft,
                hop_length=hop_length,
            )
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Seq feature extraction failed for index %d: %s", i, e)
            m = np.zeros((1, 1), dtype=np.float32)
        feature_list.append(m)
        if m.shape[0] > max_T:
            max_T = int(m.shape[0])
        if feature_dim == 0:
            feature_dim = int(m.shape[1])

    # Pad sequences to max_T
    padded = np.zeros((len(feature_list), max_T, feature_dim), dtype=np.float32)
    for i, m in enumerate(feature_list):
        T = m.shape[0]
        padded[i, :T, :] = m
    return padded, max_T, feature_dim
