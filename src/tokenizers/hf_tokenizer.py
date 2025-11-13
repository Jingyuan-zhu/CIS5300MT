"""Utilities for loading pretrained tokenizers from Hugging Face."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_TOKENIZER_ID = "Helsinki-NLP/opus-mt-en-es"
DEFAULT_TOKENIZER_DIR = Path("artifacts/tokenizers/opus_mt_en_es")


def ensure_tokenizer(
    save_dir: Path = DEFAULT_TOKENIZER_DIR,
    pretrained_id: str = DEFAULT_TOKENIZER_ID,
    force_download: bool = False,
) -> PreTrainedTokenizerBase:
    """Load the tokenizer, downloading and caching it locally if needed."""
    save_dir = Path(save_dir)
    if force_download or not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_id)
        tokenizer.save_pretrained(save_dir)
        return tokenizer

    return AutoTokenizer.from_pretrained(save_dir)


def get_helsinki_tokenizer(force_download: bool = False) -> PreTrainedTokenizerBase:
    """Convenience wrapper returning the Helsinki-NLP en-es tokenizer."""
    return ensure_tokenizer(force_download=force_download)

