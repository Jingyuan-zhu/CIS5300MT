"""Seq2Seq LSTM with attention that uses frozen Helsinki-NLP embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import MarianMTModel

from src.baselines.seq2seq_lstm_attention import (
    Seq2SeqAttentionConfig,
    Seq2SeqLSTMAttention,
)


@dataclass
class FrozenEmbeddingConfig(Seq2SeqAttentionConfig):
    """Config that mirrors Seq2SeqAttentionConfig but documents HF embedding usage."""

    hf_model_name: str = "Helsinki-NLP/opus-mt-en-es"
    freeze_embeddings: bool = True


def load_marian_embedding_matrix(
    model_name: str,
    *,
    use_decoder_embeddings: bool = False,
) -> torch.Tensor:
    """Load the embedding matrix from a pretrained Marian MT model.

    Args:
        model_name: Hugging Face model id to load.
        use_decoder_embeddings: Whether to pull decoder embeddings instead of encoder.

    Returns:
        torch.Tensor with shape (vocab_size, embed_dim).
    """
    model = MarianMTModel.from_pretrained(model_name)
    embedding_module = (
        model.model.decoder.embed_tokens if use_decoder_embeddings else model.model.encoder.embed_tokens
    )
    weight = embedding_module.weight.detach().clone().to(torch.float32)
    # Free up memory as soon as weights are copied
    del model
    torch.cuda.empty_cache()
    return weight


class Seq2SeqLSTMAttentionFrozen(Seq2SeqLSTMAttention):
    """Attention model variant that swaps in frozen HF embeddings."""

    def __init__(
        self,
        config: Seq2SeqAttentionConfig,
        pretrained_embeddings: torch.Tensor,
        *,
        freeze_embeddings: bool = True,
    ) -> None:
        super().__init__(config)
        self._replace_embeddings(pretrained_embeddings, freeze_embeddings)

    def _replace_embeddings(self, pretrained_embeddings: torch.Tensor, freeze: bool) -> None:
        if pretrained_embeddings.dim() != 2:
            raise ValueError("Pretrained embeddings must be 2D (vocab_size x embed_dim).")

        vocab_size, embed_dim = pretrained_embeddings.shape
        if vocab_size != self.config.vocab_size or embed_dim != self.config.embedding_dim:
            raise ValueError(
                "Embedding shape mismatch. "
                f"Expected ({self.config.vocab_size}, {self.config.embedding_dim}) "
                f"but got {tuple(pretrained_embeddings.shape)}."
            )

        embedding = nn.Embedding.from_pretrained(
            pretrained_embeddings,
            freeze=freeze,
            padding_idx=self.config.pad_token_id,
        )
        # Ensure module lives on same device as rest of the model
        embedding = embedding.to(self.encoder.weight_hh_l0.device)
        self.embedding = embedding


