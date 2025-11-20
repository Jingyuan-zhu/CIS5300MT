"""Generate test predictions from a saved transformer checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.translation_dataset import TranslationDataset, SequenceBatch
from src.models.transformer_model import TransformerModel
from src.tokenizers import ensure_tokenizer
from src.utils.data import load_translation_parquet, make_parallel_corpus
from src.utils.logging import setup_logger
from typing import Dict, Sequence, Tuple


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


def _pad_sequences(sequences: Sequence[torch.Tensor], pad_token_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)
    max_length = int(lengths.max().item())
    padded = torch.full((len(sequences), max_length), pad_token_id, dtype=torch.long)
    for idx, seq in enumerate(sequences):
        padded[idx, : seq.size(0)] = seq
    return padded, lengths


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


def generate_predictions(
    checkpoint_path: Path,
    test_path: Path,
    output_path: Path,
    device: str = "cuda:0",
    eval_batch_size: int = 64,
    max_source_length: int = 128,
    max_target_length: int = 128,
    max_generate_length: int = 128,
    num_workers: int = 4,
) -> None:
    """Generate test predictions from a saved checkpoint."""
    
    logger = setup_logger("generate_predictions", Path("logs"), None)
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if "config" not in checkpoint:
        raise ValueError("Checkpoint must contain 'config' key")
    
    config = checkpoint["config"]
    logger.info(f"Model config: {config}")
    
    # Setup tokenizer
    tokenizer = ensure_tokenizer(force_download=False)
    bos_id, eos_id, pad_id = get_special_tokens(tokenizer)
    logger.info(f"Tokens | BOS: {bos_id}, EOS: {eos_id}, PAD: {pad_id}")
    
    # Create model
    model = TransformerModel(config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    logger.info("Model loaded successfully")
    
    # Load test data
    logger.info(f"Loading test data from {test_path}")
    test_df = load_translation_parquet(test_path)
    test_pairs = make_parallel_corpus(test_df)
    
    test_loader = DataLoader(
        TranslationDataset(test_pairs, tokenizer, max_source_length, max_target_length),
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: collate_transformer_batch(batch, tokenizer),
    )
    
    # Generate predictions
    predictions, sources, targets = [], [], []
    
    logger.info("Generating predictions...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Generating predictions'):
            source_ids = batch.source_ids.to(device)
            source_lengths = batch.source_lengths.to(device)
            generated = model.greedy_decode(
                source_ids,
                source_lengths,
                bos_id,
                eos_id,
                max_length=max_generate_length,
                device=device,
            ).cpu()

            for idx in range(generated.size(0)):
                token_ids = generated[idx].tolist()
                if eos_id in token_ids:
                    token_ids = token_ids[: token_ids.index(eos_id)]
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                predictions.append(text)
                sources.append(batch.source_texts[idx])
                targets.append(batch.target_texts[idx])
    
    # Save predictions
    pred_df = pd.DataFrame({
        'source': sources,
        'prediction': predictions,
        'target': targets,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(output_path, index=False)
    logger.info(f"Saved test predictions to {output_path}")
    logger.info(f"Generated {len(predictions)} predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate test predictions from a saved checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file (best_model.pt)")
    parser.add_argument("--test-path", default="data/test_set.parquet", help="Path to test data")
    parser.add_argument("--output", required=True, help="Output path for predictions parquet file")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--eval-batch-size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--max-source-length", type=int, default=128, help="Max source length")
    parser.add_argument("--max-target-length", type=int, default=128, help="Max target length")
    parser.add_argument("--max-generate-length", type=int, default=128, help="Max generation length")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_predictions(
        checkpoint_path=Path(args.checkpoint),
        test_path=Path(args.test_path),
        output_path=Path(args.output),
        device=args.device,
        eval_batch_size=args.eval_batch_size,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        max_generate_length=args.max_generate_length,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

