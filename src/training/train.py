import os
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from src.data.dataset import build_dataset, CATEGORY_MAP
from src.features.mfcc import batch_extract_mfcc
from src.models.lstm import build_model

class MFCCDataset(Dataset):
    def __init__(self, X, y, sr: int, n_mfcc: int = 40):
        self.sr = sr
        self.n_mfcc = n_mfcc
        feats, _ = batch_extract_mfcc(X, sr=sr, n_mfcc=n_mfcc)
        self.X = torch.tensor(feats, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            correct += (preds == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def run_training(data_dir: str = 'data', results_dir: str = 'results', models_dir: str = 'models',
                 max_duration: float = 12.0, sr: int = 16000, n_mfcc: int = 40, batch_size: int = 32,
                 epochs: int = 20, lr: float = 1e-3, test_size: float = 0.2, val_size: float = 0.1, seed: int = 42):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    torch.manual_seed(seed)

    X, y = build_dataset(data_dir, max_duration=max_duration, sr=sr)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=(test_size + val_size), random_state=seed, stratify=y)
    relative_val = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1-relative_val, random_state=seed, stratify=y_temp)

    train_ds = MFCCDataset(X_train, y_train, sr=sr, n_mfcc=n_mfcc)
    val_ds = MFCCDataset(X_val, y_val, sr=sr, n_mfcc=n_mfcc)
    test_ds = MFCCDataset(X_test, y_test, sr=sr, n_mfcc=n_mfcc)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(n_mfcc=n_mfcc, num_classes=len(CATEGORY_MAP)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info("Model built: %s", model.__class__.__name__)

    best_val = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(1, epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(val_acc)
        logger.info("Epoch %02d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f", epoch, tr_loss, tr_acc, val_loss, val_acc)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(models_dir, 'best_lstm.pt'))
            logger.info("New best model saved (val_loss=%.4f)", val_loss)

    # Load best
    model.load_state_dict(torch.load(os.path.join(models_dir, 'best_lstm.pt'), map_location=device))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    logger.info("Test results | loss=%.4f acc=%.3f", test_loss, test_acc)

    # classification report
    model.eval()
    preds_all = []
    y_all = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds_all.extend(logits.argmax(1).cpu().numpy())
            y_all.extend(yb.numpy())
    report = classification_report(y_all, preds_all, target_names=list(CATEGORY_MAP.keys()))
    report_path = os.path.join(results_dir, 'classification_report.txt')
    # ensure report is str
    report_str = str(report)
    with open(report_path, 'w') as f:
        f.write(report_str)
    logger.info("Classification report written to %s", report_path)
    # Save history arrays explicitly to avoid typing complaints
    np.savez(os.path.join(results_dir, 'training_history.npz'),
             train_loss=np.array(history['train_loss'], dtype=np.float32),
             val_loss=np.array(history['val_loss'], dtype=np.float32),
             train_acc=np.array(history['train_acc'], dtype=np.float32),
             val_acc=np.array(history['val_acc'], dtype=np.float32))
    logger.info("Training history saved.")

    # Confusion matrix
    cm = confusion_matrix(y_all, preds_all, labels=list(range(len(CATEGORY_MAP))))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    ax[0].figure.colorbar(im, ax=ax[0])
    classes = list(CATEGORY_MAP.keys())
    ax[0].set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
              xticklabels=classes, yticklabels=classes, ylabel='True label', xlabel='Predicted label',
              title='Confusion Matrix')
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha='right')
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[0].text(j, i, format(cm[i, j], 'd'), ha='center', va='center',
                       color='white' if cm[i, j] > thresh else 'black')

    # Normalized CM
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    im2 = ax[1].imshow(cm_norm, interpolation='nearest', cmap=plt.get_cmap('Greens'))
    ax[1].figure.colorbar(im2, ax=ax[1])
    ax[1].set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
              xticklabels=classes, yticklabels=classes, ylabel='True label', xlabel='Predicted label',
              title='Confusion Matrix (Normalized)')
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha='right')
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax[1].text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center',
                       color='white' if cm_norm[i, j] > 0.5 else 'black')
    plt.tight_layout()
    cm_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", cm_path)

    # Training curves
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curves'); plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, history['train_acc'], label='Train Acc')
    plt.plot(epochs_range, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Curves'); plt.legend()
    plt.tight_layout()
    curves_path = os.path.join(results_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=150)
    plt.close()
    logger.info("Training curves saved to %s", curves_path)


if __name__ == '__main__':
    run_training()
