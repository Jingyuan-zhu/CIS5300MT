"""Evaluation utilities and CLI entry-points."""

from .metrics import corpus_bleu, corpus_chrf, comet_score, CometConfig

__all__ = ["corpus_bleu", "corpus_chrf", "comet_score", "CometConfig"]

