"""Training script for LSTM seq2seq with bidirectional encoder and attention.

This is nearly identical to train_seq2seq.py but uses the attention model.
Most code is reused through imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Import everything from vanilla training script except parse_args (we override it)
from src.baselines.train_seq2seq import (
    build_dataloader,
    get_special_tokens,
    run_epoch,
    evaluate_loss,
    evaluate_metrics,
    move_batch_to_device,
    parse_args as _parse_args_base,  # Import to reuse structure
)

# Import attention model instead of vanilla
from src.baselines.seq2seq_lstm_attention import Seq2SeqAttentionConfig, Seq2SeqLSTMAttention

# All other imports from vanilla script
import argparse
import json
from datetime import datetime
from typing import Optional

import pandas as pd
import torch
from torch.cuda.amp import GradScaler

from src.data.translation_dataset import SequenceBatch
from src.tokenizers import ensure_tokenizer
from src.utils.auto_config import auto_configure, print_hardware_info
from src.utils.data import load_translation_parquet, make_parallel_corpus, sample_dataframe
from src.utils.logging import setup_logger
from src.utils.plot_training import plot_training_history


def parse_args() -> argparse.Namespace:
    """Parse args with modified output directory for attention model."""
    parser = argparse.ArgumentParser(
        description="Train LSTM seq2seq with bidirectional encoder and attention.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Common training parameters
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=5e-4, help="Learning rate (reduced for stability)")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (auto if None)")
    parser.add_argument("--eval-batch-size", type=int, default=None, help="Eval batch size (auto if None)")
    parser.add_argument("--max-train-samples", type=int, default=200000, 
                       help="Max training samples for fair comparison (default: 200k, seed=42)")
    
    # Data paths
    parser.add_argument("--train-path", default="data/train_set.parquet", help="Training data")
    parser.add_argument("--dev-path", default="data/dev_set.parquet", help="Dev data")
    parser.add_argument("--test-path", default="data/test_set.parquet", help="Test data")
    parser.add_argument("--output-root", default="outputs/seq2seq_attention", help="Output directory")
    
    # Model architecture
    parser.add_argument("--embedding-dim", type=int, default=512, help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    
    # Auto-config and hardware
    parser.add_argument("--auto-config", action="store_true", help="Auto-detect optimal hardware settings")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="medium",
                       help="Model size for auto-config batch sizing")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu, auto if None)")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (auto if None)")
    parser.add_argument("--mixed-precision", action="store_true", default=None, help="Use mixed precision")
    
    # Other settings
    parser.add_argument("--max-source-length", type=int, default=128, help="Max source sequence length")
    parser.add_argument("--max-target-length", type=int, default=128, help="Max target sequence length")
    parser.add_argument("--max-generate-length", type=int, default=128, help="Max generation length")
    parser.add_argument("--early-stop-metric", choices=["loss", "bleu"], default="loss",
                       help="Metric for early stopping")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--force-download-tokenizer", action="store_true", help="Force re-download tokenizer")
    
    return parser.parse_args()


def train_attention() -> None:
    """Training loop for attention model - identical to vanilla except model class."""
    args = parse_args()
    
    # Apply auto-config if enabled or if hardware params not specified
    if args.auto_config or args.device is None:
        auto_cfg = auto_configure(device=args.device, model_size=args.model_size)
        if args.device is None:
            args.device = auto_cfg["device"]
        if args.batch_size is None:
            args.batch_size = auto_cfg["batch_size"]
        if args.eval_batch_size is None:
            args.eval_batch_size = auto_cfg["eval_batch_size"]
        if args.num_workers is None:
            args.num_workers = auto_cfg["num_workers"]
        if args.mixed_precision is None:
            args.mixed_precision = auto_cfg["mixed_precision"]
    
    # Set defaults if still None
    if args.batch_size is None:
        args.batch_size = 64
    if args.eval_batch_size is None:
        args.eval_batch_size = 64
    if args.num_workers is None:
        args.num_workers = 4
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mixed_precision is None:
        args.mixed_precision = False
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("seq2seq_attention", Path("logs/seq2seq_attention"), timestamp)
    logger.info("Starting seq2seq ATTENTION baseline training run.")
    logger.info(f"Run artifacts will be saved to: {run_dir}")
    
    # Print hardware info
    print("\n" + "="*60)
    print_hardware_info()
    logger.info(f"Using device: {args.device}")
    logger.info(f"Batch size: {args.batch_size} | Eval batch size: {args.eval_batch_size}")
    logger.info(f"Num workers: {args.num_workers} | Mixed precision: {args.mixed_precision}")
    print("="*60 + "\n")

    config_path = run_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    tokenizer = ensure_tokenizer(force_download=args.force_download_tokenizer)
    
    # Get special tokens (handles Marian tokenizer conventions)
    bos_id, eos_id, pad_id = get_special_tokens(tokenizer)
    logger.info(f"Special tokens: BOS={bos_id}, EOS={eos_id}, PAD={pad_id}")
    logger.info(f"Note: Marian models use EOS token for both BOS and EOS (standard convention)")
    
    # Validate tokens
    if bos_id == pad_id and eos_id == pad_id:
        logger.warning(
            "BOS, EOS, and PAD all use same token ID. This is standard for Marian models "
            "but may affect learning. The model will learn from loss signal alone."
        )

    train_df = load_translation_parquet(args.train_path)
    dev_df = load_translation_parquet(args.dev_path)
    
    # Sample training data for fair comparison across models (fixed seed)
    original_train_size = len(train_df)
    if args.max_train_samples and args.max_train_samples < original_train_size:
        train_df = sample_dataframe(train_df, args.max_train_samples, random_seed=42)
        logger.info(f"Sampled {len(train_df):,} from {original_train_size:,} training examples (seed=42)")
    else:
        logger.info(f"Using all {original_train_size:,} training examples")

    train_pairs = make_parallel_corpus(train_df)
    dev_pairs = make_parallel_corpus(dev_df)

    train_loader = build_dataloader(
        train_pairs,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_workers=args.num_workers,
        shuffle=True,
    )
    dev_loader = build_dataloader(
        dev_pairs,
        tokenizer=tokenizer,
        batch_size=args.eval_batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        num_workers=args.num_workers,
        shuffle=False,
    )

    test_loader = None
    test_exists = args.test_path and Path(args.test_path).exists()
    if test_exists:
        test_df = load_translation_parquet(args.test_path)
        test_pairs = make_parallel_corpus(test_df)
        test_loader = build_dataloader(
            test_pairs,
            tokenizer=tokenizer,
            batch_size=args.eval_batch_size,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            num_workers=args.num_workers,
            shuffle=False,
        )
        logger.info(f"Loaded test set from {args.test_path} ({len(test_pairs)} examples).")
    else:
        logger.warning(f"Test path `{args.test_path}` not found. Final evaluation on test will be skipped.")

    vocab_size_value = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size_value, int):
        vocab_size_value = len(tokenizer.get_vocab())

    # Use ATTENTION config instead of vanilla
    config = Seq2SeqAttentionConfig(
        vocab_size=vocab_size_value,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_token_id=pad_id,
    )
    
    logger.info("Using BIDIRECTIONAL encoder with BAHDANAU ATTENTION")

    device = torch.device(args.device)
    # Use attention model class
    model = Seq2SeqLSTMAttention(config).to(device)
    logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters.")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    scaler = GradScaler(enabled=args.mixed_precision)

    best_metric_value = float("inf") if args.early_stop_metric == "loss" else float("-inf")
    best_val_loss = float("inf")
    best_dev_metrics: Optional[dict] = None
    patience_counter = 0
    best_epoch = 0
    model_path = run_dir / "best_model.pt"
    
    # Training history tracking
    training_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,  # type: ignore
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            mixed_precision=args.mixed_precision,
            grad_clip=args.grad_clip,
        )

        val_loss = evaluate_loss(model, dev_loader, criterion, device)  # type: ignore
        scheduler.step(val_loss)

        dev_metrics, _, _, _ = evaluate_metrics(
            model,  # type: ignore
            dev_loader,
            tokenizer,
            device,
            bos_id,
            eos_id,
            args.max_generate_length,
            collect_predictions=False,
        )
        dev_bleu = dev_metrics["bleu"]
        dev_chrf = dev_metrics["chrf"]
        
        # Record training history
        training_history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "dev_bleu": dev_bleu,
            "dev_chrf": dev_chrf,
        })

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | dev_bleu=%.2f | dev_chrf=%.2f",
            epoch,
            train_loss,
            val_loss,
            dev_bleu,
            dev_chrf,
        )
        
        # Log sample predictions every 5 epochs
        if epoch == 1 or epoch % 5 == 0:
            sample_metrics, sample_sources, sample_preds, sample_refs = evaluate_metrics(
                model,  # type: ignore
                dev_loader,
                tokenizer,
                device,
                bos_id,
                eos_id,
                args.max_generate_length,
                collect_predictions=True,
            )
            logger.info("=" * 80)
            logger.info(f"Sample predictions at epoch {epoch}:")
            for i in range(min(3, len(sample_preds))):
                logger.info(f"  Source: {sample_sources[i][:100]}")
                logger.info(f"  Pred:   {sample_preds[i][:100]}")
                logger.info(f"  Target: {sample_refs[i][:100]}")
                logger.info(f"  Pred length: {len(sample_preds[i].split())} words")
                logger.info("-" * 80)
            logger.info("=" * 80)

        metric_value = val_loss if args.early_stop_metric == "loss" else dev_bleu
        improved = (
            metric_value < best_metric_value
            if args.early_stop_metric == "loss"
            else metric_value > best_metric_value
        )

        if improved:
            best_metric_value = metric_value
            best_val_loss = val_loss
            best_dev_metrics = {"train_loss": train_loss, "val_loss": val_loss, "bleu": dev_bleu, "chrf": dev_chrf}
            patience_counter = 0
            best_epoch = epoch
            torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, model_path)
            logger.info(
                "Improved %s to %.4f. Saved new best model checkpoint to %s",
                args.early_stop_metric,
                metric_value,
                model_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    "Early stopping triggered: no improvement in %s for %d epochs.",
                    args.early_stop_metric,
                    args.patience,
                )
                break

    if best_epoch == 0 or not model_path.exists():
        logger.error("Training finished without obtaining an improved model. Exiting.")
        return

    logger.info(
        "Training complete. Best epoch: %d | best_val_loss=%.4f | monitored=%s=%.4f",
        best_epoch,
        best_val_loss,
        args.early_stop_metric,
        best_metric_value,
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    dev_metrics_path = run_dir / f"dev_metrics_{timestamp}.json"
    if best_dev_metrics is None:
        best_dev_metrics = {"val_loss": best_val_loss}
    with dev_metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(best_dev_metrics, fh, indent=2)
    logger.info("Stored development metrics at %s", dev_metrics_path)
    
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_csv_path = run_dir / f"training_history_{timestamp}.csv"
    history_df.to_csv(history_csv_path, index=False)
    logger.info(f"Saved training history to {history_csv_path}")
    
    # Generate training plots
    try:
        plot_training_history(history_csv_path)
        logger.info(f"Generated training plot: {history_csv_path.with_suffix('.png')}")
    except Exception as e:
        logger.warning(f"Could not generate training plot: {e}")

    if test_loader is not None:
        test_metrics, sources, predictions, references = evaluate_metrics(
            model,  # type: ignore
            test_loader,
            tokenizer,
            device,
            bos_id,
            eos_id,
            args.max_generate_length,
            collect_predictions=True,
        )
        predictions_df = pd.DataFrame(
            {
                "source": list(sources),
                "prediction": list(predictions),
                "target": list(references),
            }
        )
        test_predictions_path = run_dir / f"test_predictions_{timestamp}.parquet"
        test_metrics_path = run_dir / f"test_metrics_{timestamp}.json"
        predictions_df.to_parquet(test_predictions_path, index=False)
        with test_metrics_path.open("w", encoding="utf-8") as fh:
            json.dump(test_metrics, fh, indent=2)
        logger.info("Test metrics: %s", test_metrics)
        logger.info("Saved test predictions to %s", test_predictions_path)
    else:
        logger.info("No test loader provided; skipping final evaluation.")

    logger.info("Seq2seq ATTENTION baseline run complete.")


if __name__ == "__main__":
    train_attention()

