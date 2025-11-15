"""Command-line interface for evaluating translation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from src.evaluation.metrics import CometConfig, comet_score, corpus_bleu, corpus_chrf


SUPPORTED_EXTENSIONS = {".parquet", ".csv", ".tsv", ".json", ".jsonl"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate translation outputs with BLEU, chrF, and COMET.")
    parser.add_argument("--predictions", required=True, help="Path to predictions file containing model outputs.")
    parser.add_argument("--pred-column", default="prediction", help="Column name holding model translations.")
    parser.add_argument("--ref-column", default="target", help="Column name holding reference translations.")
    parser.add_argument("--src-column", default="source", help="Column name holding source sentences.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["bleu", "chrf", "comet"],
        choices=["bleu", "chrf", "comet"],
        help="Metrics to compute.",
    )
    parser.add_argument(
        "--comet-model",
        default="Unbabel/wmt22-comet-da",
        help="COMET model checkpoint name.",
    )
    parser.add_argument("--comet-batch-size", type=int, default=8, help="Batch size for COMET scoring.")
    parser.add_argument("--comet-gpus", type=int, default=0, help="Number of GPUs to use for COMET.")
    parser.add_argument(
        "--comet-num-workers",
        type=int,
        default=1,
        help="DataLoader worker count for COMET; must be >= 1 for CPU-only runs.",
    )
    parser.add_argument("--report", help="Optional path to save metrics as JSON.")
    return parser.parse_args(argv)


def load_predictions(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported predictions file format: {path.suffix}")

    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".tsv"}:
        sep = "," if suffix == ".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unable to load predictions from: {path}")


def compute_metrics(args: argparse.Namespace) -> Dict[str, float]:
    predictions_path = Path(args.predictions)
    df = load_predictions(predictions_path)

    for column in (args.pred_column, args.ref_column, args.src_column):
        if column not in df.columns:
            raise ValueError(f"Missing required column `{column}` in predictions file.")

    predictions = df[args.pred_column].astype(str).tolist()
    references = df[args.ref_column].astype(str).tolist()
    sources = df[args.src_column].astype(str).tolist()

    scores: Dict[str, float] = {}
    if "bleu" in args.metrics:
        scores["bleu"] = corpus_bleu(predictions, references)
    if "chrf" in args.metrics:
        scores["chrf"] = corpus_chrf(predictions, references)
    if "comet" in args.metrics:
        config = CometConfig(
            model_name=args.comet_model,
            batch_size=args.comet_batch_size,
            gpus=args.comet_gpus,
            num_workers=args.comet_num_workers,
        )
        scores["comet"] = comet_score(sources, predictions, references, config=config)

    return scores


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    scores = compute_metrics(args)
    for metric, value in scores.items():
        print(f"{metric.upper()}: {value:.4f}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as fh:
            json.dump(scores, fh, indent=2)


if __name__ == "__main__":
    main()

