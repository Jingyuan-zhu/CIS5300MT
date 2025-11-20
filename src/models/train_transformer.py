"""Training script for Transformer encoder-decoder model."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict

import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

# Handle PyTorch AMP imports
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

from transformers import MarianMTModel
from src.data.translation_dataset import SequenceBatch, TranslationDataset
from src.evaluation.metrics import corpus_bleu, corpus_chrf
from src.models.transformer_model import TransformerConfig, TransformerModel
from src.tokenizers import ensure_tokenizer
from src.utils.auto_config import auto_configure, print_hardware_info
from src.utils.data import load_translation_parquet, make_parallel_corpus, sample_dataframe
from src.utils.logging import setup_logger
from src.utils.plot_training import plot_training_history


def load_pretrained_embeddings(
    model_name: str = "Helsinki-NLP/opus-mt-en-es",
    vocab_size: int = None,
    d_model: int = None,
    logger=None,
) -> Optional[torch.Tensor]:
    """Load pre-trained embedding weights from Helsinki-NLP model."""
    log = logger.info if logger else print
    log_warn = logger.warning if logger else print
    
    try:
        log(f"Loading pre-trained embeddings from {model_name}...")
        model = MarianMTModel.from_pretrained(model_name)
        embedding_module = model.model.decoder.embed_tokens
        weight = embedding_module.weight.detach().clone().to(torch.float32)
        
        if vocab_size is not None and weight.shape[0] != vocab_size:
            log_warn(f"Vocab mismatch: Pretrained {weight.shape[0]} vs Expected {vocab_size}. Skipping.")
            del model
            return None
        
        if d_model is not None and weight.shape[1] != d_model:
            log_warn(f"Dim mismatch: Pretrained {weight.shape[1]} vs Expected {d_model}. Resizing.")
            if weight.shape[1] > d_model:
                weight = weight[:, :d_model]
            else:
                padding = torch.zeros(weight.shape[0], d_model - weight.shape[1], dtype=weight.dtype)
                weight = torch.cat([weight, padding], dim=1)
        
        del model
        torch.cuda.empty_cache()
        log(f"Loaded pre-trained embeddings: {weight.shape}")
        return weight
        
    except Exception as e:
        log_warn(f"Failed to load embeddings: {e}")
        return None


def get_special_tokens(tokenizer) -> tuple[int, int, int]:
    """Get BOS, EOS, and PAD token IDs from tokenizer."""
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if pad_id is None:
        raise ValueError("Tokenizer must have a pad_token_id")
    
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    if eos_id is None:
        vocab = tokenizer.get_vocab()
        eos_id = vocab.get('</s>', vocab.get('<EOS>', pad_id))
    
    bos_id = getattr(tokenizer, 'bos_token_id', None)
    if bos_id is None:
        bos_id = eos_id 
    
    return bos_id, eos_id, pad_id


def collate_transformer_batch(
    batch: Sequence[Dict[str, torch.Tensor]],
    tokenizer,
) -> SequenceBatch:
    """Collate function ensuring correct BOS/EOS placement for Teacher Forcing."""
    bos_id, eos_id, pad_id = get_special_tokens(tokenizer)
    
    source_sequences = [item["source_ids"] for item in batch]
    target_sequences = [item["target_ids"] for item in batch]
    source_texts = [item["source_text"] for item in batch]
    target_texts = [item["target_text"] for item in batch]

    source_padded, source_lengths = _pad_sequences(source_sequences, pad_id)
    target_padded, _ = _pad_sequences(target_sequences, pad_id)
    
    batch_size, seq_len = target_padded.shape
    
    decoder_input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    decoder_input_ids[:, 0] = bos_id
    decoder_input_ids[:, 1:] = target_padded[:, :-1]
    
    decoder_target_ids = target_padded

    return SequenceBatch(
        source_ids=source_padded,
        source_lengths=source_lengths,
        decoder_input_ids=decoder_input_ids,
        decoder_target_ids=decoder_target_ids,
        source_texts=source_texts,
        target_texts=target_texts,
    )


def _pad_sequences(sequences: Sequence[torch.Tensor], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    max_length = int(lengths.max().item())
    padded = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long)
    for idx, seq in enumerate(sequences):
        padded[idx, : seq.size(0)] = seq
    return padded, lengths


def move_batch_to_device(batch: SequenceBatch, device: torch.device) -> SequenceBatch:
    return SequenceBatch(
        source_ids=batch.source_ids.to(device),
        source_lengths=batch.source_lengths.to(device),
        decoder_input_ids=batch.decoder_input_ids.to(device),
        decoder_target_ids=batch.decoder_target_ids.to(device),
        source_texts=batch.source_texts,
        target_texts=batch.target_texts,
    )


def build_dataloader(
    sentences: Sequence[Tuple[str, str]],
    tokenizer,
    batch_size: int,
    max_source_length: int,
    max_target_length: int,
    num_workers: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = TranslationDataset(sentences, tokenizer, max_source_length, max_target_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_transformer_batch(batch, tokenizer),
    )


def run_epoch(
    model: TransformerModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scaler: GradScaler,
    device: torch.device,
    mixed_precision: bool,
    grad_clip: float,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_tokens = 0
    ignore_index = int(getattr(criterion, "ignore_index", -100))
    nan_batches = 0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        
        device_type = "cuda" if "cuda" in str(device) else "cpu"
        with autocast(device_type=device_type, enabled=mixed_precision):
            tgt_len = batch.decoder_input_ids.size(1)
            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
            
            logits = model(
                source_ids=batch.source_ids,
                source_lengths=batch.source_lengths,
                decoder_input_ids=batch.decoder_input_ids,
                tgt_mask=tgt_mask,
            )
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), batch.decoder_target_ids.reshape(-1))
            
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                if nan_batches > 10: raise RuntimeError("Too many NaN losses")
                continue

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            grad_norm = clip_grad_norm_(model.parameters(), grad_clip)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                scaler.update()
                continue
        
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()

        mask = batch.decoder_target_ids != ignore_index
        tokens = torch.count_nonzero(mask).item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
        
        pbar.set_postfix({"loss": f"{total_loss / max(total_tokens, 1):.4f}"})

    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    ignore_index = int(getattr(criterion, "ignore_index", -100))

    for batch in tqdm(dataloader, desc="Eval Loss", leave=False):
        batch = move_batch_to_device(batch, device)
        tgt_len = batch.decoder_input_ids.size(1)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)
        
        logits = model(
            source_ids=batch.source_ids,
            source_lengths=batch.source_lengths,
            decoder_input_ids=batch.decoder_input_ids,
            tgt_mask=tgt_mask,
        )
        loss = criterion(logits.reshape(-1, logits.size(-1)), batch.decoder_target_ids.reshape(-1))
        mask = batch.decoder_target_ids != ignore_index
        tokens = torch.count_nonzero(mask).item()
        total_loss += loss.item() * tokens
        total_tokens += tokens

    return total_loss / max(total_tokens, 1)


def evaluate_metrics(
    model, dataloader, tokenizer, device, bos_id, eos_id, max_length, collect_predictions=False
) -> Tuple[dict, Sequence[str], Sequence[str], Sequence[str]]:
    """Used for Dev set monitoring during training."""
    model.eval()
    predictions, references, sources = [], [], []

    for batch in tqdm(dataloader, desc="Eval Metrics", leave=False):
        source_ids = batch.source_ids.to(device)
        source_lengths = batch.source_lengths.to(device)
        generated = model.greedy_decode(
            source_ids, source_lengths, bos_id, eos_id, max_length=max_length, device=device
        ).cpu()

        for idx, pred_ids in enumerate(generated):
            pred_list = pred_ids.tolist()
            if eos_id in pred_list:
                pred_list = pred_list[: pred_list.index(eos_id)]
            text = tokenizer.decode(pred_list, skip_special_tokens=True)
            predictions.append(text)
            references.append(batch.target_texts[idx])
            sources.append(batch.source_texts[idx])

    metrics = {
        "bleu": corpus_bleu(predictions, references),
        "chrf": corpus_chrf(predictions, references),
    }
    return (metrics, sources, predictions, references) if collect_predictions else (metrics, [], [], [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer encoder-decoder.")
    
    # Training Config
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=200000)
    parser.add_argument("--patience", type=int, default=5)
    
    # Data & Hardware
    parser.add_argument("--train-path", default="data/train_set.parquet")
    parser.add_argument("--dev-path", default="data/dev_set.parquet")
    parser.add_argument("--test-path", default="data/test_set.parquet")
    parser.add_argument("--output-root", default="outputs/transformer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--mixed-precision", action="store_true", default=True)
    parser.add_argument("--auto-config", action="store_true")
    parser.add_argument("--model-size", choices=["small", "medium", "large"], default="medium")
    
    # Architecture
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=6)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--dim-feedforward", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Lengths
    parser.add_argument("--max-source-length", type=int, default=128)
    parser.add_argument("--max-target-length", type=int, default=128)
    parser.add_argument("--max-generate-length", type=int, default=128)
    
    # Advanced
    parser.add_argument("--early-stop-metric", choices=["loss", "bleu"], default="loss")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--force-download-tokenizer", action="store_true")
    parser.add_argument("--use-pretrained-embeddings", action="store_true")
    parser.add_argument("--pretrained-model-name", default="Helsinki-NLP/opus-mt-en-es")
    
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    
    # Device Setup
    if args.device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.set_device(int(args.device.split(":")[-1]))
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"Using Device: {args.device}")

    # Auto Config Overrides
    if args.auto_config:
        auto_cfg = auto_configure(device=args.device, model_size=args.model_size)
        args.batch_size = args.batch_size or auto_cfg["batch_size"]
        args.eval_batch_size = args.eval_batch_size or auto_cfg["eval_batch_size"]
        args.num_workers = args.num_workers or auto_cfg["num_workers"]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.output_root) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("transformer", Path("logs/transformer"), timestamp)
    logger.info(f"Training Start. Output: {run_dir}")
    
    # Tokenizer & Data
    tokenizer = ensure_tokenizer(force_download=args.force_download_tokenizer)
    bos_id, eos_id, pad_id = get_special_tokens(tokenizer)
    logger.info(f"Tokens | BOS: {bos_id}, EOS: {eos_id}, PAD: {pad_id}")

    train_df = load_translation_parquet(args.train_path)
    dev_df = load_translation_parquet(args.dev_path)
    
    if args.max_train_samples and args.max_train_samples < len(train_df):
        train_df = sample_dataframe(train_df, args.max_train_samples, random_seed=42)
        logger.info(f"Sampled {len(train_df)} training examples")
    else:
        logger.info(f"Using full training set ({len(train_df)} examples)")

    train_loader = build_dataloader(
        make_parallel_corpus(train_df), tokenizer, args.batch_size,
        args.max_source_length, args.max_target_length, args.num_workers, shuffle=True
    )
    dev_loader = build_dataloader(
        make_parallel_corpus(dev_df), tokenizer, args.eval_batch_size,
        args.max_source_length, args.max_target_length, args.num_workers, shuffle=False
    )

    # Model Init
    vocab_size = len(tokenizer)
    pretrained_emb = None
    if args.use_pretrained_embeddings:
        pretrained_emb = load_pretrained_embeddings(
            args.pretrained_model_name, vocab_size, args.d_model, logger
        )
    else:
        logger.info("Initializing with Random Weights")

    config = TransformerConfig(
        vocab_size=vocab_size, d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward, dropout=args.dropout,
        pad_token_id=pad_id, max_seq_length=args.max_source_length,
        pretrained_embedding_weights=pretrained_emb
    )

    model = TransformerModel(config).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    ignore_idx = -100 if pad_id == eos_id else pad_id
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_idx)
    
    from torch.optim.lr_scheduler import LambdaLR
    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(100, min(4000, int(0.1 * total_steps)))
    
    def lr_lambda(step: int) -> float:
        step += 1
        return min(step**-0.5, step * warmup_steps**-1.5) * (warmup_steps**0.5)
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    logger.info(f"Scheduler: {warmup_steps} warmup steps / {total_steps} total")
    
    scaler = GradScaler(enabled=args.mixed_precision)

    best_metric = float("inf")
    history = []
    best_epoch = 0
    
    # --- TRAINING LOOP ---
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, scaler, args.device,
            args.mixed_precision, args.grad_clip, scheduler
        )
        
        val_loss = evaluate_loss(model, dev_loader, criterion, args.device)
        
        metrics, src_s, pred_s, ref_s = evaluate_metrics(
            model, dev_loader, tokenizer, args.device, bos_id, eos_id, 
            args.max_generate_length, collect_predictions=(epoch%2==0 or epoch==1)
        )
        
        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "bleu": metrics["bleu"], "chrf": metrics["chrf"]
        })
        
        logger.info(f"Epoch {epoch} | Loss: {train_loss:.4f}/{val_loss:.4f} | BLEU: {metrics['bleu']:.2f} | chrF: {metrics['chrf']:.2f}")
        
        if src_s:
            logger.info(f"Sample: {pred_s[0]}")

        # Save Best Model (Included Optimizer/Scheduler)
        if val_loss < best_metric:
            best_metric = val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch, 
                "state_dict": model.state_dict(), 
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": config, 
                "metrics": metrics
            }, run_dir / "best_model.pt")
            logger.info("Saved Best Model.")

    # --- FINAL TEST GENERATION (Matches your manual script) ---
    if args.test_path and Path(args.test_path).exists():
        logger.info(f"Loading best model (Epoch {best_epoch}) for Test Set Generation...")
        checkpoint = torch.load(run_dir / "best_model.pt", map_location=args.device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        
        logger.info(f"Loading test data from {args.test_path}...")
        test_df = load_translation_parquet(args.test_path)
        test_pairs = make_parallel_corpus(test_df)
        
        test_loader = DataLoader(
            TranslationDataset(test_pairs, tokenizer, args.max_source_length, args.max_target_length),
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=lambda batch: collate_transformer_batch(batch, tokenizer),
        )
        
        predictions, sources, targets = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Generating predictions'):
                source_ids = batch.source_ids.to(args.device)
                source_lengths = batch.source_lengths.to(args.device)
                generated = model.greedy_decode(
                    source_ids,
                    source_lengths,
                    bos_id,
                    eos_id,
                    max_length=args.max_generate_length,
                    device=args.device,
                ).cpu()

                for idx in range(generated.size(0)):
                    token_ids = generated[idx].tolist()
                    if eos_id in token_ids:
                        token_ids = token_ids[: token_ids.index(eos_id)]
                    text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    predictions.append(text)
                    sources.append(batch.source_texts[idx])
                    targets.append(batch.target_texts[idx])

        # Save Parquet for external evaluation
        pred_df = pd.DataFrame({
            'source': sources,
            'prediction': predictions,
            'target': targets,
        })
        pred_path = run_dir / 'test_predictions.parquet'
        pred_df.to_parquet(pred_path, index=False)
        logger.info(f"Saved test predictions to {pred_path}")
        logger.info("Run standard evaluation command to calculate COMET/BLEU on this file.")

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
    logger.info("Done.")

if __name__ == "__main__":
    train()