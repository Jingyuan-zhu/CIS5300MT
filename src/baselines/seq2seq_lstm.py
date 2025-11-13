"""LSTM-based encoder-decoder model for machine translation baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class Seq2SeqConfig:
    vocab_size: int
    embedding_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2
    pad_token_id: int = 0


class Seq2SeqLSTM(nn.Module):
    """A simple encoder-decoder LSTM with tied embeddings."""

    def __init__(self, config: Seq2SeqConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.encoder = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.output_projection = nn.Linear(config.hidden_dim, config.vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.output_projection.weight, -0.1, 0.1)
        nn.init.zeros_(self.output_projection.bias)

    def encode(self, source_ids: torch.Tensor, source_lengths: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        embedded = self.dropout(self.embedding(source_ids))
        packed = pack_padded_sequence(embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.encoder(packed)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        return hidden, encoder_outputs

    def decode(
        self,
        decoder_inputs: torch.Tensor,
        initial_state: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        embedded = self.dropout(self.embedding(decoder_inputs))
        decoder_outputs, _ = self.decoder(embedded, initial_state)
        logits = self.output_projection(self.dropout(decoder_outputs))
        return logits

    def forward(
        self,
        source_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        decoder_inputs: torch.Tensor,
    ) -> torch.Tensor:
        hidden, _ = self.encode(source_ids, source_lengths)
        return self.decode(decoder_inputs, hidden)

    @torch.no_grad()
    def greedy_decode(
        self,
        source_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        bos_token_id: int,
        eos_token_id: int,
        max_length: int,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        self.eval()
        device = device or source_ids.device

        hidden, _ = self.encode(source_ids, source_lengths)
        hidden = (hidden[0].to(device), hidden[1].to(device))

        batch_size = source_ids.size(0)
        inputs = torch.full((batch_size,), bos_token_id, dtype=torch.long, device=device)
        inputs = inputs.unsqueeze(1)
        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            embedded = self.embedding(inputs)
            decoder_output, hidden = self.decoder(embedded, hidden)
            logits = self.output_projection(decoder_output.squeeze(1))
            next_tokens = torch.argmax(logits, dim=-1)
            outputs.append(next_tokens)

            finished = finished | (next_tokens == eos_token_id)
            if finished.all():
                break

            inputs = next_tokens.unsqueeze(1)

        if outputs:
            predictions = torch.stack(outputs, dim=1)
        else:
            predictions = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        return predictions

