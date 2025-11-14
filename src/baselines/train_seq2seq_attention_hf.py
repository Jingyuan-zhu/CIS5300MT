"""Training script for LSTM seq2seq with attention using frozen HF embeddings."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from torch.cuda.amp import GradScaler

from src.baselines.train_seq2seq import (
    build_dataloader,
    evaluate_loss,
    evaluate_metrics,
    get_special_tokens,
    run_epoch,
)
from src.baselines.seq2seq_lstm_attention import Seq2SeqAttentionConfig
from src.baselines.seq2seq_lstm_attention_hf import (
    Seq2SeqLSTMAttentionFrozen,
    load_marian_embedding_matrix,
)
from src.tokenizers import ensure_tokenizer
from src.utils.auto_config import auto_configure, print_hardware_info
from src.utils.data import load_translation_parquet, make_parallel_corpus, sample_dataframe
from src.utils.logging import setup_logger
from src.utils.plot_training import plot_training_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train LSTM seq2seq + attention with frozen Helsinki-NLP embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=200000)

    parser.add_argument("--train-path", default="data/train_set.parquet")
    parser.add_argument("--dev-path", default="data/dev_set.parquet")
    parser.add_argument("--test-path", default="data/test_set.parquet")
    parser.add_argument("--output-root", default="outputs/seq2seq_attention_hfemb")

    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--auto-config", action="store_true")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="medium")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--mixed-precision", action="store_true", default=None)

    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-generate-length", type=int, default=128)
    parser.add_argument("--early-stop-metric", choices=["loss", "bleu"], default="loss")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--force-download-tokenizer", action="store_true")

    parser.add_argument("--hf-model-name", default="Helsinki-NLP/opus-mt-en-es")

    return parser.parse_args()


def train_attention_hf() -> None:
    args = parse_args()

    if args.auto_config or args.device is None:
        auto_cfg = auto_configure(device=args.device, model_size=args.model_size)
        args.device = args.device or auto_cfg["device"]
        args.batch_size = args.batch_size or auto_cfg["batch_size"]
        args.eval_batch_size = args.eval_batch_size or auto_cfg["eval_batch_size"]
        args.num_workers = args.num_workers or auto_cfg["num_workers"]
        args.mixed_precision = auto_cfg["mixed_precision"] if args.mixed_precision is None else args.mixed_precision

    args.batch_size = args.batch_size or 64
    args.eval_batch_size = args.eval_batch_size or 64
    args.num_workers = args.num_workers or 4
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.mixed_precision = bool(args.mixed_precision)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("seq2seq_attention_hfemb", Path("logs/seq2seq_attention_hfemb"), timestamp)
    logger.info("Starting seq2seq ATTENTION (HF embeddings) training run.")
    logger.info(f"Run artifacts will be saved to: {run_dir}")

    print("\n" + "=" * 60)
    print_hardware_info()
    logger.info(f"Using device: {args.device}")
    logger.info(f"Batch size: {args.batch_size} | Eval batch size: {args.eval_batch_size}")
    logger.info(f"Num workers: {args.num_workers} | Mixed precision: {args.mixed_precision}")
    print("=" * 60 + "\n")

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=2)

    tokenizer = ensure_tokenizer(force_download=args.force_download_tokenizer)
    bos_id, eos_id, pad_id = get_special_tokens(tokenizer)
    logger.info(f"Special tokens: BOS={bos_id}, EOS={eos_id}, PAD={pad_id}")

    train_df = load_translation_parquet(args.train_path)
    dev_df = load_translation_parquet(args.dev_path)

    if args.max_train_samples and args.max_train_samples < len(train_df):
        train_df = sample_dataframe(train_df, args.max_train_samples, random_seed=42)
        logger.info(f"Sampled {len(train_df):,} from training data (seed=42)")
    else:
        logger.info(f"Using all {len(train_df):,} training examples")

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
    if args.test_path and Path(args.test_path).exists():
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

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if not isinstance(vocab_size, int):
        vocab_size = len(tokenizer.get_vocab())

    logger.info(f"Loading pretrained embeddings from {args.hf_model_name} ...")
    embedding_weight = load_marian_embedding_matrix(args.hf_model_name)
    if embedding_weight.size(0) != vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({vocab_size}) does not match embedding matrix ({embedding_weight.size(0)})."
        )

    config = Seq2SeqAttentionConfig(
        vocab_size=vocab_size,
        embedding_dim=embedding_weight.size(1),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_token_id=pad_id,
    )

    device = torch.device(args.device)
    model = Seq2SeqLSTMAttentionFrozen(
        config,
        embedding_weight,
        freeze_embeddings=True,
    ).to(device)
    logger.info(
        "Model initialized with %.2fM parameters (embeddings frozen).",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
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

    training_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,  # type: ignore[arg-type]
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
            mixed_precision=args.mixed_precision,
            grad_clip=args.grad_clip,
        )

        val_loss = evaluate_loss(model, dev_loader, criterion, device)  # type: ignore[arg-type]
        scheduler.step(val_loss)

        dev_metrics, _, _, _ = evaluate_metrics(
            model,  # type: ignore[arg-type]
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
        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "dev_bleu": dev_bleu,
                "dev_chrf": dev_chrf,
            }
        )

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | dev_bleu=%.2f | dev_chrf=%.2f",
            epoch,
            train_loss,
            val_loss,
            dev_bleu,
            dev_chrf,
        )

        if epoch == 1 or epoch % 5 == 0:
            sample_metrics, sample_sources, sample_preds, sample_refs = evaluate_metrics(
                model,  # type: ignore[arg-type]
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
        improved = metric_value < best_metric_value if args.early_stop_metric == "loss" else metric_value > best_metric_value

        if improved:
            best_metric_value = metric_value
            best_val_loss = val_loss
            best_dev_metrics = dev_metrics
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
            logger.info("Improved metric to %.4f. Saved new best model.", metric_value)
        else:
            patience_counter += 1
            logger.info("No improvement. Patience %d/%d", patience_counter, args.patience)

        if patience_counter >= args.patience:
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    if training_history:
        import pandas as pd

        history_df = pd.DataFrame(training_history)
        history_path = run_dir / "training_history.csv"
        history_df.to_csv(history_path, index=False)
        logger.info(f"Saved training history to {history_path}")
        plot_training_history(history_path)

    if best_dev_metrics is not None:
        logger.info("Best epoch %d | val_loss=%.4f | dev BLEU=%.2f | dev chrF=%.2f", best_epoch, best_val_loss, best_dev_metrics["bleu"], best_dev_metrics["chrf"])
    else:
        logger.warning("No best dev metrics recorded.")

    if best_dev_metrics is None:
        logger.warning("Skipping final evaluation because no best checkpoint was saved.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))

    if test_loader is not None:
        test_metrics, sources, preds, refs = evaluate_metrics(
            model,  # type: ignore[arg-type]
            test_loader,
            tokenizer,
            device,
            bos_id,
            eos_id,
            args.max_generate_length,
            collect_predictions=True,
        )
        logger.info("Test metrics: %s", test_metrics)

        predictions_df = __build_predictions_df(sources, preds, refs)
        test_predictions_path = run_dir / "test_predictions.parquet"
        predictions_df.to_parquet(test_predictions_path, index=False)
        logger.info(f"Saved test predictions to {test_predictions_path}")

        run_eval_metrics(
            run_dir,
            test_predictions_path,
            args.device,
        )
    else:
        logger.warning("Test loader not available, skipping test evaluation.")


def __build_predictions_df(sources: list[str], preds: list[str], refs: list[str]):
    import pandas as pd

    return pd.DataFrame({"source": sources, "prediction": preds, "target": refs})


def run_eval_metrics(run_dir: Path, predictions_path: Path, device: str) -> None:
    import subprocess

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metrics_path = run_dir / f"test_metrics_{timestamp}.json"

    eval_cmd = [
        "python",
        "-m",
        "src.evaluation.run_evaluation",
        "--predictions",
        str(predictions_path),
        "--metrics",
        "bleu",
        "chrf",
        "comet",
        "--comet-num-workers",
        "1",
        "--comet-gpus",
        "1" if device.startswith("cuda") else "-1",
        "--report",
        str(metrics_path),
    ]

    try:
        subprocess.run(eval_cmd, check=True)
        print(f"Saved comprehensive metrics to {metrics_path}")
    except subprocess.CalledProcessError as exc:
        print(f"Evaluation script failed: {exc}")


if __name__ == "__main__":
    train_attention_hf()


