"""Utility functions for plotting training history."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments


def plot_training_history(
    history_csv: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Plot training history from CSV file.
    
    Parameters
    ----------
    history_csv:
        Path to the training_history_*.csv file.
    output_path:
        Where to save the plot. If None, saves next to CSV with .png extension.
    figsize:
        Figure size in inches (width, height).
    """
    history_path = Path(history_csv)
    if not history_path.exists():
        raise FileNotFoundError(f"History file not found: {history_csv}")
    
    # Load history
    df = pd.read_csv(history_path)
    
    # Set output path
    if output_path is None:
        output_path = history_path.with_suffix('.png')
    else:
        output_path = Path(output_path)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot 1: Loss curves
    ax1 = axes[0]
    ax1.plot(df['epoch'], df['train_loss'], 'o-', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(df['epoch'], df['val_loss'], 's-', label='Val Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: BLEU score
    ax2 = axes[1]
    ax2.plot(df['epoch'], df['dev_bleu'], 'o-', label='Dev BLEU', 
             color='green', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('BLEU Score', fontsize=12)
    ax2.set_title('Development Set BLEU Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add summary statistics
    best_epoch = df.loc[df['dev_bleu'].idxmax(), 'epoch']
    best_bleu = df['dev_bleu'].max()
    final_train_loss = df['train_loss'].iloc[-1]
    final_val_loss = df['val_loss'].iloc[-1]
    
    fig.text(0.5, 0.02, 
             f'Best: Epoch {best_epoch:.0f}, BLEU {best_bleu:.2f} | '
             f'Final: Train Loss {final_train_loss:.4f}, Val Loss {final_val_loss:.4f}',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Training history plot saved to: {output_path}")


def plot_multiple_runs(
    csv_files: list[Union[str, Path]],
    labels: list[str],
    output_path: Union[str, Path],
    metric: str = "dev_bleu",
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """Compare multiple training runs on the same plot.
    
    Parameters
    ----------
    csv_files:
        List of paths to training_history_*.csv files.
    labels:
        List of labels for each run (e.g., model names).
    output_path:
        Where to save the comparison plot.
    metric:
        Metric to plot: "dev_bleu", "train_loss", "val_loss", or "dev_chrf".
    figsize:
        Figure size in inches (width, height).
    """
    if len(csv_files) != len(labels):
        raise ValueError("Number of CSV files must match number of labels")
    
    plt.figure(figsize=figsize)
    
    for csv_file, label in zip(csv_files, labels):
        df = pd.read_csv(csv_file)
        plt.plot(df['epoch'], df[metric], 'o-', label=label, linewidth=2, markersize=5)
    
    plt.xlabel('Epoch', fontsize=12)
    
    # Set appropriate y-label
    if 'loss' in metric:
        plt.ylabel('Loss', fontsize=12)
        title = f'Comparison: {metric.replace("_", " ").title()}'
    elif 'bleu' in metric:
        plt.ylabel('BLEU Score', fontsize=12)
        title = 'Comparison: Development BLEU Score'
    else:
        plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
        title = f'Comparison: {metric.replace("_", " ").title()}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_path)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comparison plot saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Plot training history from CSV")
    parser.add_argument("history_csv", help="Path to training_history_*.csv file")
    parser.add_argument("--output", "-o", help="Output plot path (default: same as CSV with .png)")
    parser.add_argument("--figsize", type=int, nargs=2, default=[12, 8],
                       help="Figure size in inches (width height)")
    
    args = parser.parse_args()
    
    plot_training_history(
        args.history_csv,
        output_path=args.output,
        figsize=tuple(args.figsize)
    )

