"""Run Helsinki-NLP/opus-mt-en-es directly on test set as ultimate baseline.

This script loads the pretrained Marian MT model and evaluates it on the test set
without any fine-tuning. This represents the upper bound performance anchor.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

from src.evaluation.metrics import corpus_bleu, corpus_chrf
from src.utils.data import load_translation_parquet, make_parallel_corpus
from src.utils.logging import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretrained Helsinki-NLP/opus-mt-en-es on test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--test-path", default="data/test_set.parquet", help="Test data path")
    parser.add_argument("--model-name", default="Helsinki-NLP/opus-mt-en-es", 
                       help="HuggingFace model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for inference")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on")
    parser.add_argument("--output-dir", default="outputs/hf_baseline", help="Output directory")
    
    return parser.parse_args()


def batch_translate(
    texts: list[str],
    model: MarianMTModel,
    tokenizer: MarianTokenizer,
    batch_size: int,
    max_length: int,
    device: str,
) -> list[str]:
    """Translate texts in batches."""
    translations = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Translating"):
        batch = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        # Generate
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                max_length=max_length,
                num_beams=4,  # Use beam search for better quality
                early_stopping=True,
            )
        
        # Decode
        batch_translations = tokenizer.batch_decode(translated, skip_special_tokens=True)
        translations.extend(batch_translations)
    
    return translations


def main():
    args = parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("hf_baseline", Path("logs/hf_baseline"), timestamp)
    logger.info(f"Starting HF baseline evaluation with {args.model_name}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = MarianTokenizer.from_pretrained(args.model_name)
    model = MarianMTModel.from_pretrained(args.model_name).to(args.device)
    model.eval()
    
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
    
    # Load test data
    logger.info(f"Loading test data from {args.test_path}")
    test_df = load_translation_parquet(args.test_path)
    test_pairs = make_parallel_corpus(test_df)
    
    sources = [pair[0] for pair in test_pairs]
    references = [pair[1] for pair in test_pairs]
    
    logger.info(f"Loaded {len(sources)} test examples")
    
    # Translate
    logger.info("Translating...")
    predictions = batch_translate(
        sources,
        model,
        tokenizer,
        args.batch_size,
        args.max_length,
        args.device,
    )
    
    # Evaluate
    logger.info("Computing metrics...")
    bleu = corpus_bleu(predictions, references)
    chrf = corpus_chrf(predictions, references)
    
    metrics = {
        "bleu": bleu,
        "chrf": chrf,
        "model": args.model_name,
        "test_size": len(predictions),
        "timestamp": timestamp,
    }
    
    logger.info(f"BLEU: {bleu:.4f}")
    logger.info(f"CHRF: {chrf:.4f}")
    
    # Save results
    predictions_df = pd.DataFrame({
        "source": sources,
        "prediction": predictions,
        "target": references,
    })
    
    predictions_path = output_dir / f"predictions_{timestamp}.parquet"
    metrics_path = output_dir / f"metrics_{timestamp}.json"
    
    predictions_df.to_parquet(predictions_path, index=False)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Saved predictions to {predictions_path}")
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Print sample predictions
    logger.info("\n" + "=" * 80)
    logger.info("Sample predictions:")
    for i in range(min(5, len(predictions))):
        logger.info(f"\nExample {i+1}:")
        logger.info(f"  Source: {sources[i][:100]}")
        logger.info(f"  Pred:   {predictions[i][:100]}")
        logger.info(f"  Target: {references[i][:100]}")
    logger.info("=" * 80)
    
    logger.info("HF baseline evaluation complete!")
    
    print(f"\n{'='*60}")
    print(f"Helsinki-NLP/opus-mt-en-es Results:")
    print(f"  BLEU:  {bleu:.4f}")
    print(f"  CHRF:  {chrf:.4f}")
    print(f"  Model: {args.model_name}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

