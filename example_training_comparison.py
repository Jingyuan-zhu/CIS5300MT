"""Example: Compare training runs across different models.

This shows how to use the plotting utility to compare multiple models.
"""

from pathlib import Path
from src.utils.plot_training import plot_multiple_runs

# Example: Compare three different runs
run_dirs = [
    "outputs/seq2seq_lstm/20251113-022146",
    "outputs/seq2seq_lstm/20251113-123456",  # Example timestamp
    "outputs/seq2seq_lstm/20251113-234567",  # Example timestamp
]

# Find training history CSV files
csv_files = []
for run_dir in run_dirs:
    history_files = list(Path(run_dir).glob("training_history_*.csv"))
    if history_files:
        csv_files.append(history_files[0])

if len(csv_files) >= 2:
    # Compare BLEU scores
    plot_multiple_runs(
        csv_files=csv_files,
        labels=["200k samples", "500k samples", "1M samples"],
        output_path="outputs/comparison_bleu.png",
        metric="dev_bleu"
    )
    
    # Compare validation loss
    plot_multiple_runs(
        csv_files=csv_files,
        labels=["200k samples", "500k samples", "1M samples"],
        output_path="outputs/comparison_val_loss.png",
        metric="val_loss"
    )
    
    print("✓ Created comparison plots!")
else:
    print("Need at least 2 training runs to compare.")
    print("Run training multiple times with different settings:")
    print("  python -m src.baselines.train_seq2seq --auto-config --max-train-samples 200000")
    print("  python -m src.baselines.train_seq2seq --auto-config --max-train-samples 500000")

