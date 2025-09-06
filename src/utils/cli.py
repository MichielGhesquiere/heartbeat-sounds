import argparse

from src.training.train import run_experiment_suite, run_training
from src.utils.logging_config import setup_logging
from src.utils.visualize import visualize_examples


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Heart sound classification (PyTorch)")
    subparsers = parser.add_subparsers(dest="command", required=False)

    # Train subcommand
    train_p = subparsers.add_parser("train", help="Train model")
    train_p.add_argument("--data_dir", type=str, default="data")
    train_p.add_argument("--results_dir", type=str, default="results")
    train_p.add_argument("--models_dir", type=str, default="models")
    train_p.add_argument("--max_duration", type=float, default=12.0)
    train_p.add_argument("--sr", type=int, default=16000)
    train_p.add_argument("--n_mfcc", type=int, default=40)
    train_p.add_argument("--batch_size", type=int, default=32)
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--test_size", type=float, default=0.2)
    train_p.add_argument("--val_size", type=float, default=0.1)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument(
        "--feature_type", type=str, default="sequence", choices=["mfcc", "sequence"]
    )
    train_p.add_argument("--seq_mode", type=str, default="seq_mel_mfcc")
    train_p.add_argument("--n_mels", type=int, default=64)
    train_p.add_argument("--no_deltas", action="store_true")
    train_p.add_argument("--no_stats", action="store_true")
    train_p.add_argument("--no_class_weighting", action="store_true")
    train_p.add_argument("--focal", action="store_true")
    train_p.add_argument("--gamma", type=float, default=2.0)
    train_p.add_argument("--experiment_label", type=str, default=None)
    train_p.add_argument(
        "--suite", action="store_true", help="Run predefined experiment suite for incremental gains"
    )

    # Visualize subcommand
    vis_p = subparsers.add_parser("visualize", help="Create waveform and spectrogram images")
    vis_p.add_argument("--data_dir", type=str, default="data")
    vis_p.add_argument("--results_dir", type=str, default="results")
    vis_p.add_argument("--max_duration", type=float, default=12.0)
    vis_p.add_argument("--sr", type=int, default=16000)
    vis_p.add_argument("--per_class", type=int, default=2)
    vis_p.add_argument("--n_mels", type=int, default=128)
    vis_p.add_argument("--no_audio", action="store_true", help="Do not save audio snippets")

    args = parser.parse_args()

    if args.command == "visualize":
        visualize_examples(
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            max_duration=args.max_duration,
            sr=args.sr,
            per_class=args.per_class,
            n_mels=args.n_mels,
            save_audio=not args.no_audio,
        )
    else:
        # default to training if no command provided
        if getattr(args, "suite", False):
            run_experiment_suite(
                data_dir=args.data_dir,
                results_dir=args.results_dir,
                models_dir=args.models_dir,
                max_duration=args.max_duration,
                sr=args.sr,
                n_mfcc=args.n_mfcc,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                test_size=args.test_size,
                val_size=args.val_size,
                seed=args.seed,
                seq_mode=args.seq_mode,
                n_mels=args.n_mels,
            )
        else:
            run_training(
                data_dir=args.data_dir,
                results_dir=args.results_dir,
                models_dir=args.models_dir,
                max_duration=args.max_duration,
                sr=args.sr,
                n_mfcc=args.n_mfcc,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                test_size=args.test_size,
                val_size=args.val_size,
                seed=args.seed,
                feature_type=args.feature_type,
                seq_mode=args.seq_mode,
                n_mels=args.n_mels,
                add_deltas=not args.no_deltas,
                add_stats=not args.no_stats,
                class_weighting=not args.no_class_weighting,
                focal_loss=args.focal,
                gamma=args.gamma,
                experiment_label=args.experiment_label,
            )


if __name__ == "__main__":
    main()
