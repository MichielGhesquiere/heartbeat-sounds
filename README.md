# Heartbeat Sounds - PyTorch Pipeline

Refactored the exploratory notebook into a modular, reproducible PyTorch pipeline.

## Project Layout

```
├── data/                # (ignored by git - raw data)
├── src/
│   ├── data/dataset.py  # Loading & assembling dataset
│   ├── features/mfcc.py # MFCC feature extraction helpers
│   ├── models/lstm.py   # PyTorch LSTM model
│   ├── training/train.py# Training script & loops
│   ├── utils/cli.py     # CLI entrypoint (train / visualize)
│   └── utils/visualize.py # Visualization helpers
├── models/              # Saved model weights (*.pt)
├── results/             # Reports & metrics + visualizations
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Run Training

```bash
python -m src.utils.cli train --data_dir data --epochs 25 --batch_size 32
```

Artifacts:
- `models/best_lstm.pt` best checkpoint
- `results/classification_report.txt` metrics
- `results/training_history.npz` arrays (loss/accuracy)

## Visualize Sample Waveforms & Spectrograms
Export a few examples per class (default 2) including waveform PNG, mel spectrogram PNG, and audio copy.

```bash
python -m src.utils.cli visualize --data_dir data --per_class 3 --results_dir results
```
Disable writing audio snippets:
```bash
python -m src.utils.cli visualize --no_audio
```
Outputs go to:
```
results/visualizations/
  waveforms/*.png
  spectrograms/*.png
  audio/*.wav
```

## Extending
- Add augmentations in `dataset.py`
- Swap model architecture in `models/`
- Add new feature extractors under `features/`
- Add confusion matrix plotting script

## Notes
Current class map: artifact=0, murmur=1, normal=2.

Adjust `CATEGORY_MAP` in `dataset.py` for more classes.
