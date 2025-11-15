"""Metric helpers for machine translation evaluation."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Union

import sacrebleu

try:
    from comet import download_model, load_from_checkpoint
except ImportError:  # pragma: no cover - optional dependency
    download_model = None
    load_from_checkpoint = None


def corpus_bleu(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Compute corpus-level BLEU using sacrebleu.
    
    Note: sacrebleu expects references as List[List[str]] where outer list is examples
    and inner list is alternative references.
    """
    # Validate inputs
    if len(predictions) == 0 or len(references) == 0:
        return 0.0
    if len(predictions) != len(references):
        raise ValueError(f"Prediction count ({len(predictions)}) != reference count ({len(references)})")
    
    # sacrebleu expects: predictions as List[str], references as List[List[str]]
    result = sacrebleu.corpus_bleu(predictions, [references])
    return float(result.score)


def corpus_chrf(predictions: Sequence[str], references: Sequence[str], char_order: int = 6) -> float:
    """Compute corpus-level chrF score.
    
    Note: sacrebleu expects references as List[List[str]] where outer list is examples
    and inner list is alternative references.
    """
    # Validate inputs
    if len(predictions) == 0 or len(references) == 0:
        return 0.0
    if len(predictions) != len(references):
        raise ValueError(f"Prediction count ({len(predictions)}) != reference count ({len(references)})")
    
    # sacrebleu expects: predictions as List[str], references as List[List[str]]
    result = sacrebleu.corpus_chrf(predictions, [references], char_order=char_order)
    return float(result.score)


@dataclass
class CometConfig:
    model_name: str = "Unbabel/wmt22-comet-da"
    batch_size: int = 8
    gpus: Optional[int] = 0
    devices: Optional[Union[int, Sequence[int]]] = None
    num_workers: int = 0
    progress_bar: bool = True


def comet_score(
    sources: Sequence[str],
    predictions: Sequence[str],
    references: Sequence[str],
    config: Optional[CometConfig] = None,
) -> float:
    """Compute COMET score for the provided system outputs."""
    if download_model is None or load_from_checkpoint is None:
        raise ImportError(
            "COMET is not installed. Install `unbabel-comet` to enable this metric."
        )

    cfg = config or CometConfig()
    model_path = download_model(cfg.model_name)
    model = load_from_checkpoint(model_path)
    samples = [{"src": src, "mt": hyp, "ref": ref} for src, hyp, ref in zip(sources, predictions, references)]
    if cfg.num_workers == 0 and hasattr(model, "multiprocessing_context"):
        setattr(model, "multiprocessing_context", None)

    predict_signature = inspect.signature(model.predict)
    predict_kwargs = {}
    params = predict_signature.parameters
    if "batch_size" in params:
        predict_kwargs["batch_size"] = cfg.batch_size
    if "gpus" in params and cfg.gpus is not None:
        predict_kwargs["gpus"] = cfg.gpus
    if "devices" in params and cfg.devices is not None:
        predict_kwargs["devices"] = cfg.devices
    if "num_workers" in params:
        predict_kwargs["num_workers"] = cfg.num_workers
    if "progress_bar" in params:
        predict_kwargs["progress_bar"] = cfg.progress_bar
    elif "show_progress_bar" in params:
        predict_kwargs["show_progress_bar"] = cfg.progress_bar
    output = model.predict(samples, **predict_kwargs)
    return float(output["system_score"])

