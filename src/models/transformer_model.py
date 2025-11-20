"""Transformer encoder-decoder model for machine translation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    pad_token_id: int = 0
    max_seq_length: int = 128
    pretrained_embedding_weights: Optional[torch.Tensor] = None  # Optional: (vocab_size, d_model)


class TransformerModel(nn.Module):
    """Transformer encoder-decoder for sequence-to-sequence translation."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        # Embeddings
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout, config.max_seq_length)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.num_decoder_layers)

        # Output projection (weight-tied with embedding for parameter efficiency)
        # Note: We use bias=False because we'll tie weights with embedding
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights (or use pre-trained embeddings)
        use_pretrained = config.pretrained_embedding_weights is not None
        if use_pretrained:
            # Initialize with pre-trained embeddings
            if config.pretrained_embedding_weights.shape != (config.vocab_size, config.d_model):
                raise ValueError(
                    f"Pre-trained embedding shape {config.pretrained_embedding_weights.shape} "
                    f"does not match expected ({config.vocab_size}, {config.d_model})"
                )
            self.embedding.weight.data.copy_(config.pretrained_embedding_weights)
            # Ensure padding token is still zero
            if config.pad_token_id is not None:
                self.embedding.weight.data[config.pad_token_id].zero_()
        else:
            # Random initialization
            self._init_weights()
        
        # Weight tying: share embedding and output projection weights
        # This reduces parameters and is standard practice in NMT
        # The output projection uses the same weight matrix as embeddings (transposed)
        self.output_projection.weight = self.embedding.weight

    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        # Use Xavier uniform for better initialization
        nn.init.xavier_uniform_(self.embedding.weight)
        # Set padding token embedding to zero
        if self.config.pad_token_id is not None:
            self.embedding.weight.data[self.config.pad_token_id].zero_()
        # Note: output_projection.weight is tied to embedding.weight, so no separate init needed

    def forward(
        self,
        source_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            source_ids: (batch, src_len) source token IDs
            source_lengths: (batch,) source sequence lengths
            decoder_input_ids: (batch, tgt_len) decoder input token IDs
            source_mask: (batch, src_len) source padding mask (True = valid, False = pad)
            tgt_mask: (batch, tgt_len, tgt_len) causal mask for decoder

        Returns:
            logits: (batch, tgt_len, vocab_size) output logits
        """
        # Create source padding mask if not provided
        if source_mask is None:
            source_mask = self._create_padding_mask(source_ids, source_lengths)

        # Embed and add positional encoding
        # Scale embeddings by sqrt(d_model) before adding positional encodings
        # This is required for standard Transformer architecture and MarianMT compatibility
        src_emb = self.dropout(self.embedding(source_ids))
        src_emb = src_emb * math.sqrt(self.config.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.dropout(self.embedding(decoder_input_ids))
        tgt_emb = tgt_emb * math.sqrt(self.config.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)

        # Encode
        memory = self.encoder(src_emb, src_key_padding_mask=~source_mask)

        # Decode
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=~source_mask,
        )

        # Project to vocabulary
        logits = self.output_projection(output)
        return logits

    def _create_padding_mask(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Create padding mask from token IDs and lengths.

        Args:
            ids: (batch, seq_len) token IDs
            lengths: (batch,) sequence lengths

        Returns:
            mask: (batch, seq_len) True for valid tokens, False for padding
        """
        batch_size, seq_len = ids.shape
        mask = torch.arange(seq_len, device=ids.device).unsqueeze(0) < lengths.unsqueeze(1)
        return mask

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
        """Greedy decoding.

        Args:
            source_ids: (batch, src_len) source token IDs
            source_lengths: (batch,) source sequence lengths
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            max_length: Maximum generation length
            device: Device to run on

        Returns:
            generated_ids: (batch, gen_len) generated token IDs
        """
        self.eval()
        device = device or source_ids.device
        batch_size = source_ids.size(0)

        # Create source mask
        source_mask = self._create_padding_mask(source_ids, source_lengths)

        # Encode source
        # Apply same scaling as in forward pass to match training behavior
        src_emb = self.dropout(self.embedding(source_ids) * math.sqrt(self.config.d_model))
        src_emb = self.pos_encoder(src_emb)
        memory = self.encoder(src_emb, src_key_padding_mask=~source_mask)

        # Initialize decoder input with BOS (for internal use only)
        decoder_input = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device,
        )
        
        # Track which sequences are finished
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Store generated tokens (excluding BOS, consistent with LSTM baselines)
        outputs = []

        # Generate tokens
        for step in range(max_length):
            # Create causal mask for current sequence length
            tgt_len = decoder_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=device)

            # Embed and add positional encoding
            # Apply same scaling as in forward pass to match training behavior
            tgt_emb = self.dropout(self.embedding(decoder_input) * math.sqrt(self.config.d_model))
            tgt_emb = self.pos_encoder(tgt_emb)

            # Decode
            output = self.decoder(
                tgt_emb,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=~source_mask,
            )

            # Get next token logits
            next_token_logits = self.output_projection(output[:, -1, :])  # (batch, vocab_size)
            
            # Removed manual EOS masking heuristic - let the model learn naturally
            next_token_id = next_token_logits.argmax(dim=-1)  # (batch,)

            # Store generated token (excluding BOS)
            outputs.append(next_token_id)

            # Update finished sequences (allow stopping after minimum 2 tokens)
            if step >= 1:
                finished = finished | (next_token_id == eos_token_id)
            
            # Stop if all sequences are finished (but only after minimum length of 2 tokens)
            if step >= 1 and finished.all():
                break
            
            # Append to decoder input for next iteration
            decoder_input = torch.cat([decoder_input, next_token_id.unsqueeze(1)], dim=1)

        # Stack outputs (consistent with LSTM: returns [token1, token2, ...] without BOS)
        if outputs:
            predictions = torch.stack(outputs, dim=1)
        else:
            predictions = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        return predictions


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model) input embeddings

        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

