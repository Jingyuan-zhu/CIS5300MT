"""Utility to inspect model predictions and diagnose issues."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def inspect_predictions(predictions_file: str | Path, n_samples: int = 10) -> None:
    """Inspect predictions to diagnose model issues.
    
    Parameters
    ----------
    predictions_file:
        Path to predictions parquet file.
    n_samples:
        Number of samples to display.
    """
    df = pd.read_parquet(predictions_file)
    
    print("=" * 80)
    print(f"Inspecting: {predictions_file}")
    print(f"Total examples: {len(df)}")
    print("=" * 80)
    
    # Compute statistics
    pred_lengths = df['prediction'].str.split().str.len()
    target_lengths = df['target'].str.split().str.len()
    
    print("\n📊 Length Statistics:")
    print(f"  Prediction - Mean: {pred_lengths.mean():.1f}, Median: {pred_lengths.median():.1f}, "
          f"Min: {pred_lengths.min()}, Max: {pred_lengths.max()}")
    print(f"  Target     - Mean: {target_lengths.mean():.1f}, Median: {target_lengths.median():.1f}, "
          f"Min: {target_lengths.min()}, Max: {target_lengths.max()}")
    
    # Check for issues
    print("\n🔍 Potential Issues:")
    
    # Empty predictions
    empty_preds = (df['prediction'] == '').sum()
    if empty_preds > 0:
        print(f"  ⚠️  {empty_preds} empty predictions ({empty_preds/len(df)*100:.1f}%)")
    
    # Very short predictions
    very_short = (pred_lengths <= 2).sum()
    if very_short > len(df) * 0.1:
        print(f"  ⚠️  {very_short} predictions with ≤2 words ({very_short/len(df)*100:.1f}%)")
    
    # Check for repetition
    def has_repetition(text: str) -> bool:
        words = text.split()
        if len(words) < 3:
            return False
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True
        return False
    
    repetitive = df['prediction'].apply(has_repetition).sum()
    if repetitive > len(df) * 0.05:
        print(f"  ⚠️  {repetitive} predictions with word repetition ({repetitive/len(df)*100:.1f}%)")
    
    # Check for only punctuation/special chars
    only_special = df['prediction'].str.match(r'^[^\w\s]+$').sum()
    if only_special > 0:
        print(f"  ⚠️  {only_special} predictions with only punctuation/special chars")
    
    print("\n📝 Sample Predictions:")
    print("=" * 80)
    
    # Show diverse samples
    indices = [0, len(df)//4, len(df)//2, 3*len(df)//4, len(df)-1]
    indices = [i for i in indices if i < len(df)][:n_samples]
    
    for idx in indices:
        row = df.iloc[idx]
        print(f"\nExample {idx+1}:")
        print(f"  Source:     {row['source'][:100]}")
        print(f"  Prediction: {row['prediction'][:100]}")
        print(f"  Target:     {row['target'][:100]}")
        print(f"  Pred words: {len(row['prediction'].split())}, Target words: {len(row['target'].split())}")
        print("-" * 80)
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect model predictions")
    parser.add_argument("predictions_file", help="Path to predictions parquet file")
    parser.add_argument("-n", "--n-samples", type=int, default=10, help="Number of samples to show")
    
    args = parser.parse_args()
    inspect_predictions(args.predictions_file, args.n_samples)

