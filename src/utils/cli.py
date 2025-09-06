import argparse
from src.training.train import run_training
from src.utils.visualize import visualize_examples
from src.utils.logging_config import setup_logging

def main():
    setup_logging()
    parser = argparse.ArgumentParser(description='Heart sound classification (PyTorch)')
    subparsers = parser.add_subparsers(dest='command', required=False)

    # Train subcommand
    train_p = subparsers.add_parser('train', help='Train model')
    train_p.add_argument('--data_dir', type=str, default='data')
    train_p.add_argument('--results_dir', type=str, default='results')
    train_p.add_argument('--models_dir', type=str, default='models')
    train_p.add_argument('--max_duration', type=float, default=12.0)
    train_p.add_argument('--sr', type=int, default=16000)
    train_p.add_argument('--n_mfcc', type=int, default=40)
    train_p.add_argument('--batch_size', type=int, default=32)
    train_p.add_argument('--epochs', type=int, default=20)
    train_p.add_argument('--lr', type=float, default=1e-3)
    train_p.add_argument('--test_size', type=float, default=0.2)
    train_p.add_argument('--val_size', type=float, default=0.1)
    train_p.add_argument('--seed', type=int, default=42)

    # Visualize subcommand
    vis_p = subparsers.add_parser('visualize', help='Create waveform and spectrogram images')
    vis_p.add_argument('--data_dir', type=str, default='data')
    vis_p.add_argument('--results_dir', type=str, default='results')
    vis_p.add_argument('--max_duration', type=float, default=12.0)
    vis_p.add_argument('--sr', type=int, default=16000)
    vis_p.add_argument('--per_class', type=int, default=2)
    vis_p.add_argument('--n_mels', type=int, default=128)
    vis_p.add_argument('--no_audio', action='store_true', help='Do not save audio snippets')

    args = parser.parse_args()

    if args.command == 'visualize':
        visualize_examples(data_dir=args.data_dir,
                           results_dir=args.results_dir,
                           max_duration=args.max_duration,
                           sr=args.sr,
                           per_class=args.per_class,
                           n_mels=args.n_mels,
                           save_audio=not args.no_audio)
    else:
        # default to training if no command provided
        run_training(data_dir=args.data_dir,
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
                     seed=args.seed)

if __name__ == '__main__':
    main()
