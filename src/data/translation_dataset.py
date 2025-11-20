"""PyTorch dataset and collate utilities for translation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class SequenceBatch:
    source_ids: torch.Tensor
    source_lengths: torch.Tensor
    decoder_input_ids: torch.Tensor
    decoder_target_ids: torch.Tensor
    source_texts: List[str]
    target_texts: List[str]


class TranslationDataset(Dataset[Dict[str, torch.Tensor]]):
    """Wraps a pandas DataFrame of translations for tokenization on the fly."""

    def __init__(
        self,
        sentences: Sequence[Tuple[str, str]],
        tokenizer: PreTrainedTokenizerBase,
        max_source_length: int = 128,
        max_target_length: int = 128,
    ) -> None:
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_text, target_text = self.sentences[index]
        source_ids = self._encode(source_text, self.max_source_length)
        target_ids = self._encode(target_text, self.max_target_length)
        return {
            "source_ids": torch.tensor(source_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "source_text": source_text,
            "target_text": target_text,
        }

    def _encode(self, text: str, max_length: int) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )


def collate_translation_batch(
    batch: Sequence[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> SequenceBatch:
    """Pad and assemble a batch for training.
    
    For Marian tokenizers, BOS and EOS are the same token (typically 0).
    The tokenizer only adds EOS at the end, so we need to prepend BOS to decoder_input.
    
    Target format from tokenizer: [token1, token2, ..., tokenN, EOS]
    Decoder input should be: [BOS, token1, token2, ..., tokenN]  
    Decoder target should be: [token1, token2, ..., tokenN, EOS]
    """
    source_sequences = [item["source_ids"] for item in batch]
    target_sequences = [item["target_ids"] for item in batch]
    source_texts = [item["source_text"] for item in batch]
    target_texts = [item["target_text"] for item in batch]

    source_padded, source_lengths = _pad_sequences(source_sequences, pad_token_id)
    target_padded, _ = _pad_sequences(target_sequences, pad_token_id)

    # Check if target already starts with BOS (for backward compatibility)
    # Marian tokenizers typically don't add BOS at start, only EOS at end
    bos_token_id = 0  # Marian uses 0 for both BOS and EOS
    first_tokens = target_padded[:, 0]
    has_bos = (first_tokens == bos_token_id).all()
    
    if has_bos:
        # Target already has BOS: [BOS, token1, ..., tokenN, EOS]
        # decoder_input = [BOS, token1, ..., tokenN]
        # decoder_target = [token1, ..., tokenN, EOS]
        decoder_input_ids = target_padded[:, :-1]
        decoder_target_ids = target_padded[:, 1:]
    else:
        # Target doesn't have BOS: [token1, ..., tokenN, EOS]
        # decoder_input should be: [BOS, token1, ..., tokenN]
        # decoder_target should be: [token1, ..., tokenN, EOS]
        # 
        # Teacher forcing: at step i, predict decoder_target[i] given decoder_input[:i+1]
        # Step 0: predict token1 given [BOS] -> decoder_target[0] = token1
        # Step 1: predict token2 given [BOS, token1] -> decoder_target[1] = token2
        # Step N: predict EOS given [BOS, token1, ..., tokenN] -> decoder_target[N] = EOS
        batch_size, seq_len = target_padded.shape
        decoder_input_ids = torch.full((batch_size, seq_len), pad_token_id, dtype=torch.long)
        decoder_input_ids[:, 0] = bos_token_id  # First token is BOS
        decoder_input_ids[:, 1:] = target_padded[:, :-1]  # [token1, ..., tokenN] (exclude EOS)
        decoder_target_ids = target_padded  # [token1, ..., tokenN, EOS] (full target)

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

