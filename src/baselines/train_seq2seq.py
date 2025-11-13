"""Training script for the LSTM-based seq2seq translation baseline."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.baselines.seq2seq_lstm import Seq2SeqConfig, Seq2SeqLSTM
from src.data.translation_dataset import SequenceBatch, TranslationDataset, collate_translation_batch
from src.evaluation.metrics import corpus_bleu, corpus_chrf
from src.tokenizers import ensure_tokenizer
from src.utils.auto_config import auto_configure, print_hardware_info
from src.utils.data import load_translation_parquet, make_parallel_corpus, sample_dataframe
from src.utils.logging import setup_logger
from src.utils.plot_training import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple LSTM seq2seq baseline with auto-config.",
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
    parser.add_argument("--output-root", default="outputs/seq2seq_lstm", help="Output directory")
    
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


def build_dataloader(
    sentences: Sequence[Tuple[str, str]],
    tokenizer,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TranslationDataset(
        sentences,
        tokenizer=tokenizer,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_translation_batch(batch, tokenizer.pad_token_id),
    )


def get_special_tokens(tokenizer) -> tuple[int, int, int]:
    """Get BOS, EOS, and PAD token IDs from tokenizer.
    
    For Marian models (Helsinki-NLP), the convention is:
    - Uses EOS token (typically 0) for both sequence start and end
    - PAD token is usually also 0
    
    Returns:
        (bos_id, eos_id, pad_id)
    """
    # Get PAD token
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None:
        raise ValueError("Tokenizer must have a pad_token_id")
    
    # For Marian models, try to get EOS token first
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    
    # If no EOS token, try other alternatives
    if eos_id is None:
        # Try to find </s> token
        vocab = tokenizer.get_vocab()
        if '</s>' in vocab:
            eos_id = vocab['</s>']
        elif '<EOS>' in vocab:
            eos_id = vocab['<EOS>']
        else:
            # Last resort: use pad_id
            eos_id = pad_id
    
    # For Marian models, BOS = EOS (they use the same token)
    # This is the standard Marian convention
    bos_id = eos_id
    
    return bos_id, eos_id, pad_id


def train() -> None:
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

    logger = setup_logger("seq2seq_lstm", Path("logs/seq2seq_lstm"), timestamp)
    logger.info("Starting seq2seq baseline training run.")
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

    test_loader: Optional[DataLoader] = None
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

    config = Seq2SeqConfig(
        vocab_size=vocab_size_value,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_token_id=pad_id,
    )

    device = torch.device(args.device)
    model = Seq2SeqLSTM(config).to(device)
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
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            mixed_precision=args.mixed_precision,
            grad_clip=args.grad_clip,
        )

        val_loss = evaluate_loss(model, dev_loader, criterion, device)
        scheduler.step(val_loss)

        dev_metrics, _, _, _ = evaluate_metrics(
            model,
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
        
        # Log sample predictions every epoch for debugging
        if epoch == 1 or epoch % 5 == 0:
            sample_metrics, sample_sources, sample_preds, sample_refs = evaluate_metrics(
                model,
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
        # Should not happen, but guard.
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
            model,
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

    logger.info("Seq2seq baseline run complete.")


def run_epoch(
    model: Seq2SeqLSTM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: GradScaler,
    device: torch.device,
    mixed_precision: bool,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    ignore_index = int(getattr(criterion, "ignore_index", -100))

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=mixed_precision):
            logits = model(batch.source_ids, batch.source_lengths, batch.decoder_input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch.decoder_target_ids.reshape(-1))

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        mask = batch.decoder_target_ids != ignore_index
        tokens = torch.count_nonzero(mask).item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        
        pbar.set_postfix({"loss": f"{total_loss / max(total_tokens, 1):.4f}"})

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(
    model: Seq2SeqLSTM,
    dataloader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    ignore_index = int(getattr(criterion, "ignore_index", -100))

    pbar = tqdm(dataloader, desc="Evaluating loss", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        logits = model(batch.source_ids, batch.source_lengths, batch.decoder_input_ids)
        loss = criterion(logits.reshape(-1, logits.size(-1)), batch.decoder_target_ids.reshape(-1))
        mask = batch.decoder_target_ids != ignore_index
        tokens = torch.count_nonzero(mask).item()
        total_loss += loss.item() * tokens
        total_tokens += tokens

    return total_loss / max(total_tokens, 1)


def evaluate_metrics(
    model: Seq2SeqLSTM,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    bos_token_id: int,
    eos_token_id: int,
    max_length: int,
    collect_predictions: bool = False,
) -> Tuple[dict, Sequence[str], Sequence[str], Sequence[str]]:
    model.eval()
    predictions = []
    references = []
    sources = []

    pbar = tqdm(dataloader, desc="Generating predictions", leave=False)
    for batch in pbar:
        source_ids = batch.source_ids.to(device)
        source_lengths = batch.source_lengths.to(device)
        generated = model.greedy_decode(
            source_ids,
            source_lengths,
            bos_token_id,
            eos_token_id,
            max_length=max_length,
            device=device,
        ).cpu()

        for idx, pred_ids in enumerate(generated):
            pred_list = pred_ids.tolist()
            if eos_token_id in pred_list:
                pred_list = pred_list[: pred_list.index(eos_token_id)]
            text = tokenizer.decode(pred_list, skip_special_tokens=True)
            predictions.append(text)
            references.append(batch.target_texts[idx])
            sources.append(batch.source_texts[idx])

    metrics = {
        "bleu": corpus_bleu(predictions, references),
        "chrf": corpus_chrf(predictions, references),
    }

    if collect_predictions:
        return metrics, sources, predictions, references

    return metrics, [], [], []


def move_batch_to_device(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    return SequenceBatch(
        source_ids=batch.source_ids.to(device),
        source_lengths=batch.source_lengths.to(device),
        decoder_input_ids=batch.decoder_input_ids.to(device),
        decoder_target_ids=batch.decoder_target_ids.to(device),
        source_texts=batch.source_texts,
        target_texts=batch.target_texts,
    )


if __name__ == "__main__":
    train()

