import logging
import os

import matplotlib

matplotlib.use("Agg")  # headless BEFORE importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from src.data.dataset import CATEGORY_MAP, build_dataset
from src.features.mfcc import batch_extract_mfcc
from src.features.sequential import batch_extract_sequence
from src.models.lstm import build_model, build_sequence_model

logger = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        mode: str,
        sr: int,
        n_mfcc: int = 40,
        seq_mode: str = "seq_mel_mfcc",
        n_mels: int = 64,
        add_deltas: bool = True,
        add_stats: bool = True,
    ):
        self.mode = mode  # 'mfcc' or 'sequence'
        if mode == "mfcc":
            feats, _ = batch_extract_mfcc(X, sr=sr, n_mfcc=n_mfcc)
            self.X = torch.tensor(feats, dtype=torch.float32)
        else:
            seq_feats, _, feature_dim = batch_extract_sequence(
                X,
                sr=sr,
                mode=seq_mode,
                n_mels=n_mels,
                n_mfcc=n_mfcc,
                add_deltas=add_deltas,
                add_spectral_stats=add_stats,
            )
            self.X = torch.tensor(seq_feats, dtype=torch.float32)
            self.feature_dim = feature_dim
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


def run_training(
    data_dir: str = "data",
    results_dir: str = "results",
    models_dir: str = "models",
    max_duration: float = 12.0,
    sr: int = 16000,
    n_mfcc: int = 40,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 1e-3,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    feature_type: str = "sequence",
    seq_mode: str = "seq_mel_mfcc",
    n_mels: int = 64,
    add_deltas: bool = True,
    add_stats: bool = True,
    class_weighting: bool = True,
    focal_loss: bool = False,
    gamma: float = 2.0,
    save_experiment: bool = True,
    experiment_label: str | None = None,
):
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    # Derive experiment label if not provided
    if experiment_label is None:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_label = f"{feature_type}_{seq_mode}_cw{int(class_weighting)}_d{int(add_deltas)}_s{int(add_stats)}_{timestamp}"
    # Per-experiment directories
    exp_results_dir = os.path.join(results_dir, experiment_label)
    exp_models_dir = os.path.join(models_dir, experiment_label)
    if save_experiment:
        os.makedirs(exp_results_dir, exist_ok=True)
        os.makedirs(exp_models_dir, exist_ok=True)
    torch.manual_seed(seed)

    X, y = build_dataset(data_dir, max_duration=max_duration, sr=sr)

    # --- Split into train/val/test with graceful fallback for tiny synthetic datasets ---
    primary_test_fraction = test_size + val_size
    if primary_test_fraction >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X,
            y,
            test_size=primary_test_fraction,
            random_state=seed,
            stratify=y if len(np.unique(y)) == len(CATEGORY_MAP) else None,
        )
    except ValueError:
        # Fallback: no stratify
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=primary_test_fraction, random_state=seed, stratify=None
        )

    # If val_size very small or dataset too small for a second stratified split, fallback.
    relative_val = val_size / (test_size + val_size) if (test_size + val_size) > 0 else 0.0
    perform_val_split = (
        val_size > 0 and len(y_temp) >= 4
    )  # need at least 4 samples to hope for stratification
    if perform_val_split:
        try:
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp,
                y_temp,
                test_size=1 - relative_val,
                random_state=seed,
                stratify=y_temp if len(np.unique(y_temp)) == len(CATEGORY_MAP) else None,
            )
        except ValueError:
            logger.warning(
                "Validation split skipped (dataset too small for stratified split). Using training set for validation metrics."
            )
            X_val, y_val = X_train, y_train
            X_test, y_test = X_temp, y_temp
    else:
        if val_size > 0:
            logger.warning(
                "Validation split disabled due to insufficient samples (len(y_temp)=%d).",
                len(y_temp),
            )
        X_val, y_val = X_train, y_train
        X_test, y_test = X_temp, y_temp

    train_ds = FeatureDataset(
        X_train,
        y_train,
        mode="mfcc" if feature_type == "mfcc" else "sequence",
        sr=sr,
        n_mfcc=n_mfcc,
        seq_mode=seq_mode,
        n_mels=n_mels,
        add_deltas=add_deltas,
        add_stats=add_stats,
    )
    val_ds = FeatureDataset(
        X_val,
        y_val,
        mode="mfcc" if feature_type == "mfcc" else "sequence",
        sr=sr,
        n_mfcc=n_mfcc,
        seq_mode=seq_mode,
        n_mels=n_mels,
        add_deltas=add_deltas,
        add_stats=add_stats,
    )
    test_ds = FeatureDataset(
        X_test,
        y_test,
        mode="mfcc" if feature_type == "mfcc" else "sequence",
        sr=sr,
        n_mfcc=n_mfcc,
        seq_mode=seq_mode,
        n_mels=n_mels,
        add_deltas=add_deltas,
        add_stats=add_stats,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if feature_type == "mfcc":
        model = build_model(n_mfcc=n_mfcc, num_classes=len(CATEGORY_MAP)).to(device)
    else:
        feature_dim = train_ds.feature_dim
        model = build_sequence_model(feature_dim=feature_dim, num_classes=len(CATEGORY_MAP)).to(
            device
        )

    # Class weighting
    if class_weighting:
        class_counts = np.bincount(y_train, minlength=len(CATEGORY_MAP))
        inv_freq = class_counts.max() / (class_counts + 1e-6)
        weights = torch.tensor(
            inv_freq / inv_freq.sum() * len(CATEGORY_MAP), dtype=torch.float32, device=device
        )
    else:
        weights = None

    if focal_loss:
        # Implement simple focal loss wrapper
        class FocalLoss(nn.Module):
            def __init__(self, gamma: float = 2.0, weight=None):
                super().__init__()
                self.gamma = gamma
                self.weight = weight
                self.ce = nn.CrossEntropyLoss(weight=weight)

            def forward(self, logits, targets):
                logpt = -self.ce(logits, targets)
                pt = torch.exp(logpt)
                loss = -((1 - pt) ** self.gamma) * logpt
                return loss

        criterion = FocalLoss(gamma=gamma, weight=weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logger.info("Model built: %s", model.__class__.__name__)

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        logger.info(
            "Epoch %02d | train_loss=%.4f acc=%.3f | val_loss=%.4f acc=%.3f",
            epoch,
            tr_loss,
            tr_acc,
            val_loss,
            val_acc,
        )
        if val_loss < best_val:
            best_val = val_loss
            if save_experiment:
                torch.save(model.state_dict(), os.path.join(exp_models_dir, "best_model.pt"))
            else:
                torch.save(model.state_dict(), os.path.join(models_dir, "best_model.pt"))
            logger.info("New best model saved (val_loss=%.4f)", val_loss)

    # Load best
    best_model_load_path = os.path.join(
        exp_models_dir if save_experiment else models_dir, "best_model.pt"
    )
    model.load_state_dict(torch.load(best_model_load_path, map_location=device))
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
    # Always include all defined classes (even if some missing in tiny splits) to keep downstream
    # artifact generation stable during unit tests / synthetic datasets.
    all_labels = list(range(len(CATEGORY_MAP)))
    report = classification_report(  # type: ignore[assignment]
        y_all,
        preds_all,
        labels=all_labels,
        target_names=list(CATEGORY_MAP.keys()),
        output_dict=True,
        zero_division=0,
    )
    report_path = os.path.join(
        exp_results_dir if save_experiment else results_dir, "classification_report.txt"
    )
    # ensure report is str
    # Save nice text and JSON
    import json

    rep_dict = report  # type: ignore
    if save_experiment:
        with open(report_path, "w") as f:
            for cls, stats in rep_dict.items():  # type: ignore[attr-defined]
                if isinstance(stats, dict) and "precision" in stats:
                    f.write(
                        f"{cls}: precision={stats['precision']:.3f} recall={stats['recall']:.3f} f1={stats['f1-score']:.3f} support={stats['support']}\n"
                    )
                else:
                    f.write(f"{cls}: {stats}\n")
        with open(os.path.join(results_dir, "classification_report.json"), "w") as jf:
            json.dump(report, jf, indent=2)
    logger.info("Classification report written to %s", report_path)
    # Save history arrays explicitly to avoid typing complaints
    np.savez(
        os.path.join(exp_results_dir if save_experiment else results_dir, "training_history.npz"),
        train_loss=np.array(history["train_loss"], dtype=np.float32),
        val_loss=np.array(history["val_loss"], dtype=np.float32),
        train_acc=np.array(history["train_acc"], dtype=np.float32),
        val_acc=np.array(history["val_acc"], dtype=np.float32),
    )
    logger.info("Training history saved.")

    # Confusion matrix
    cm = confusion_matrix(y_all, preds_all, labels=list(range(len(CATEGORY_MAP))))
    # Track per-class recall improvements across experiments: append to CSV
    if save_experiment:
        # classification_report output is Dict[str, Dict[str, float]]; use .get with default
        per_class_recalls = [
            float(rep_dict.get(name, {}).get("recall", float("nan")))  # type: ignore[call-arg]
            for name in CATEGORY_MAP.keys()
        ]
        # Build label
        if experiment_label is None:
            experiment_label = f"{feature_type}-{'cw' if class_weighting else 'nocw'}-{'stats' if add_stats else 'nostats'}-{'deltas' if add_deltas else 'nodeltas'}"
        exp_csv = os.path.join(
            results_dir, "experiment_recalls_detailed.csv"
        )  # keep global aggregation at root
        header = (
            "experiment_label,feature_type,seq_mode,class_weighting,focal_loss,add_stats,add_deltas,"
            + ",".join([f"recall_{k}" for k in CATEGORY_MAP.keys()])
            + "\n"
        )
        line = (
            f"{experiment_label},{feature_type},{seq_mode},{int(class_weighting)},{int(focal_loss)},{int(add_stats)},{int(add_deltas)},"
            + ",".join(f"{r:.4f}" for r in per_class_recalls)
            + "\n"
        )
        if not os.path.exists(exp_csv):
            with open(exp_csv, "w") as ef:
                ef.write(header)
                ef.write(line)
        else:
            with open(exp_csv, "a") as ef:
                ef.write(line)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    ax[0].figure.colorbar(im, ax=ax[0])
    classes = list(CATEGORY_MAP.keys())
    ax[0].set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax[0].text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    # Normalized CM
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    im2 = ax[1].imshow(cm_norm, interpolation="nearest", cmap=plt.get_cmap("Greens"))
    ax[1].figure.colorbar(im2, ax=ax[1])
    ax[1].set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix (Normalized)",
    )
    plt.setp(ax[1].get_xticklabels(), rotation=45, ha="right")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax[1].text(
                j,
                i,
                f"{cm_norm[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    plt.tight_layout()
    cm_path = os.path.join(
        exp_results_dir if save_experiment else results_dir, "confusion_matrix.png"
    )
    plt.savefig(cm_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", cm_path)

    # Training curves
    epochs_range = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Acc")
    plt.plot(epochs_range, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.tight_layout()
    curves_path = os.path.join(
        exp_results_dir if save_experiment else results_dir, "training_curves.png"
    )
    plt.savefig(curves_path, dpi=150)
    plt.close()
    logger.info("Training curves saved to %s", curves_path)

    logger.info("Artifacts stored in %s", exp_results_dir if save_experiment else results_dir)
    return {
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "report": rep_dict,
        "experiment_label": experiment_label,
        "feature_type": feature_type,
        "results_dir": exp_results_dir if save_experiment else results_dir,
        "models_dir": exp_models_dir if save_experiment else models_dir,
    }


def plot_experiment_recalls(results_dir: str = "results"):
    csv_path = os.path.join(results_dir, "experiment_recalls_detailed.csv")
    if not os.path.exists(csv_path):
        logger.warning("No experiment recall CSV found at %s", csv_path)
        return
    # Read manually
    with open(csv_path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    header = lines[0].split(",")
    rows = [line.split(",") for line in lines[1:]]
    # Indices
    label_idx = header.index("experiment_label")
    class_keys = list(CATEGORY_MAP.keys())
    recall_indices = {cls: header.index(f"recall_{cls}") for cls in class_keys}
    exp_labels = [r[label_idx] for r in rows]
    recalls = {cls: [float(r[recall_indices[cls]]) for r in rows] for cls in class_keys}
    plt.figure(figsize=(10, 5))
    for cls, series in recalls.items():
        plt.plot(range(1, len(series) + 1), series, marker="o", label=cls)
    plt.xticks(range(1, len(exp_labels) + 1), exp_labels, rotation=30, ha="right")
    plt.ylabel("Recall")
    plt.xlabel("Experiment")
    plt.ylim(0, 1.05)
    plt.title("Per-Class Recall Across Experiments")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(results_dir, "recall_progress.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Experiment recall progression plot saved to %s", out_path)


def run_experiment_suite(
    data_dir: str = "data", results_dir: str = "results", models_dir: str = "models", **common
):
    """Run a predefined series of experiments to quantify incremental gains.

    Steps:
      1. Baseline MFCC (mean pooled) no weighting.
      2. Sequence (log-mel+mfcc) no weighting, no deltas/stats.
      3. + Class weighting.
      4. + Deltas & spectral stats.
    """
    experiments = [
        dict(
            feature_type="mfcc",
            class_weighting=False,
            add_deltas=False,
            add_stats=False,
            experiment_label="1_mfcc_baseline",
        ),
        dict(
            feature_type="sequence",
            class_weighting=False,
            add_deltas=False,
            add_stats=False,
            experiment_label="2_seq_base",
        ),
        dict(
            feature_type="sequence",
            class_weighting=True,
            add_deltas=False,
            add_stats=False,
            experiment_label="3_seq_cw",
        ),
        dict(
            feature_type="sequence",
            class_weighting=True,
            add_deltas=True,
            add_stats=True,
            experiment_label="4_seq_cw_deltas_stats",
        ),
    ]
    results = []
    for cfg in experiments:
        logger.info("Running experiment %s", cfg["experiment_label"])
        res = run_training(  # type: ignore[arg-type]
            data_dir=data_dir,
            results_dir=results_dir,
            models_dir=models_dir,
            max_duration=common.get("max_duration", 12.0),
            sr=common.get("sr", 16000),
            n_mfcc=common.get("n_mfcc", 40),
            batch_size=common.get("batch_size", 32),
            epochs=common.get("epochs", 15),
            lr=common.get("lr", 1e-3),
            test_size=common.get("test_size", 0.2),
            val_size=common.get("val_size", 0.1),
            seed=common.get("seed", 42),
            seq_mode=common.get("seq_mode", "seq_mel_mfcc"),
            n_mels=common.get("n_mels", 64),
            gamma=common.get("gamma", 2.0),
            feature_type=cfg["feature_type"],  # type: ignore[index]
            class_weighting=cfg["class_weighting"],  # type: ignore[index]
            add_deltas=cfg["add_deltas"],  # type: ignore[index]
            add_stats=cfg["add_stats"],  # type: ignore[index]
            focal_loss=cfg.get("focal_loss", False),  # type: ignore[arg-type]
            experiment_label=cfg["experiment_label"],  # type: ignore[index]
            save_experiment=True,
        )
        results.append(res)
    plot_experiment_recalls(results_dir=results_dir)
    # Additional incremental gains visualization (delta vs first experiment)
    try:
        csv_path = os.path.join(results_dir, "experiment_recalls_detailed.csv")
        if os.path.exists(csv_path):
            import csv

            with open(csv_path) as f:
                reader = list(csv.reader(f))
            header = reader[0]
            rows = reader[1:]
            if rows:
                base = rows[0]
                labels = [r[0] for r in rows]
                class_keys = list(CATEGORY_MAP.keys())
                indices = {cls: header.index(f"recall_{cls}") for cls in class_keys}
                deltas = {
                    cls: [float(r[indices[cls]]) - float(base[indices[cls]]) for r in rows]
                    for cls in class_keys
                }
                plt.figure(figsize=(10, 5))
                for cls, series in deltas.items():
                    plt.plot(range(1, len(series) + 1), series, marker="o", label=cls)
                plt.axhline(0, color="gray", linestyle="--", linewidth=1)
                plt.xticks(range(1, len(labels) + 1), labels, rotation=30, ha="right")
                plt.ylabel("Recall Improvement vs Baseline")
                plt.xlabel("Experiment")
                plt.title("Incremental Recall Gains")
                plt.legend()
                plt.tight_layout()
                gain_path = os.path.join(results_dir, "recall_gains.png")
                plt.savefig(gain_path, dpi=150)
                plt.close()
                logger.info("Incremental recall gains plot saved to %s", gain_path)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to create recall gains plot: %s", e)
    return results


if __name__ == "__main__":
    run_training()
