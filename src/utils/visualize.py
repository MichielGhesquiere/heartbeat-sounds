import os
from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import logging

logger = logging.getLogger(__name__)

from src.data.dataset import build_dataset, CATEGORY_MAP


def _ensure_dirs(base_results: str):
    vis_root = os.path.join(base_results, 'visualizations')
    paths = {
        'root': vis_root,
        'waveforms': os.path.join(vis_root, 'waveforms'),
        'spectrograms': os.path.join(vis_root, 'spectrograms'),
        'audio': os.path.join(vis_root, 'audio')
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _to_float32(y: np.ndarray) -> np.ndarray:
    """Ensure audio is float32 in [-1, 1]. Handles common PCM integer formats."""
    if not np.issubdtype(y.dtype, np.floating):
        orig_dtype = y.dtype
        y = y.astype(np.float32)
        # scale typical PCM ranges
        if orig_dtype == np.int16:
            y /= 32768.0
        elif orig_dtype == np.int32:
            y /= 2147483648.0
        else:
            # generic normalization if amplitudes look outside [-1,1]
            max_abs = np.max(np.abs(y)) or 1.0
            if max_abs > 1:
                y /= max_abs
    return y.astype(np.float32)


def plot_waveform(y: np.ndarray, sr: int, out_path: str, title: str):
    y = _to_float32(y)
    plt.figure(figsize=(10, 2))
    times = np.arange(len(y)) / sr
    plt.plot(times, y, linewidth=0.8)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_mel_spectrogram(y: np.ndarray, sr: int, out_path: str, title: str, n_mels: int = 128):
    y = _to_float32(y)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def visualize_examples(data_dir: str = 'data', results_dir: str = 'results', max_duration: float = 12.0,
                       sr: int = 16000, per_class: int = 2, n_mels: int = 128, save_audio: bool = True):
    """Create waveform and mel-spectrogram images (and optional audio copies) for a few examples per class.

    Args:
        data_dir: Directory containing set_a / set_b folders.
        results_dir: Base results directory where visualizations will be written.
        max_duration: Max duration used when constructing dataset (for consistent padding).
        sr: Sample rate.
        per_class: Number of examples per class to export.
        n_mels: Mel filter bank size.
        save_audio: If True, saves the (possibly padded) audio to wav files.
    """
    paths = _ensure_dirs(results_dir)
    X, y = build_dataset(data_dir, max_duration=max_duration, sr=sr)

    # reverse label map
    int_to_label = {v: k for k, v in CATEGORY_MAP.items()}

    # group indices per class
    class_indices: Dict[int, List[int]] = {lbl: [] for lbl in int_to_label.keys()}
    for idx, lbl in enumerate(y):
        if len(class_indices[lbl]) < per_class:
            class_indices[lbl].append(idx)
        # early exit if all collected
    if all(len(v) >= per_class for v in class_indices.values()):
        pass

    for lbl, indices in class_indices.items():
        label_name = int_to_label[lbl]
        for j, i in enumerate(indices):
            y_audio = X[i]
            base_name = f"{label_name}_{j}".replace(' ', '_')
            wave_out = os.path.join(paths['waveforms'], f"waveform_{base_name}.png")
            spec_out = os.path.join(paths['spectrograms'], f"mel_{base_name}.png")
            plot_waveform(y_audio, sr, wave_out, f"Waveform - {label_name} #{j}")
            plot_mel_spectrogram(y_audio, sr, spec_out, f"Mel Spectrogram - {label_name} #{j}", n_mels=n_mels)
            if save_audio:
                # Ensure proper numeric dtype (avoid object arrays from container)
                try:
                    y_safe = np.asarray(y_audio)
                    if y_safe.dtype == object:
                        # flatten any nested object elements
                        y_safe = np.concatenate([np.asarray(x).ravel() for x in y_safe])
                    y_safe = _to_float32(y_safe.squeeze())
                    audio_out = os.path.join(paths['audio'], f"{base_name}.wav")
                    sf.write(audio_out, y_safe, sr)
                except Exception as wav_err:
                    print(f"Skipping audio write for {base_name}: {wav_err}")

    logger.info("Saved visualizations to: %s", paths['root'])


if __name__ == '__main__':
    visualize_examples()
