"""LSTM encoder-decoder with bidirectional encoder and Bahdanau attention."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class Seq2SeqAttentionConfig:
    vocab_size: int
    embedding_dim: int = 512
    hidden_dim: int = 512
    num_layers: int = 2
    dropout: float = 0.2
    pad_token_id: int = 0


class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.energy_proj = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(
        self, 
        query: torch.Tensor,  # (batch, hidden_dim)
        encoder_outputs: torch.Tensor,  # (batch, src_len, hidden_dim)
        src_mask: Optional[torch.Tensor] = None,  # (batch, src_len)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute attention context and weights.
        
        Args:
            query: Decoder hidden state
            encoder_outputs: All encoder hidden states
            src_mask: Mask for padding positions (1 = valid, 0 = pad)
        
        Returns:
            context: Weighted sum of encoder outputs
            attn_weights: Attention distribution over source
        """
        # Project query and keys
        query_proj = self.query_proj(query).unsqueeze(1)  # (batch, 1, hidden)
        key_proj = self.key_proj(encoder_outputs)  # (batch, src_len, hidden)
        
        # Compute attention energies
        energy = self.energy_proj(torch.tanh(query_proj + key_proj)).squeeze(-1)  # (batch, src_len)
        
        # Apply mask if provided
        if src_mask is not None:
            energy = energy.masked_fill(src_mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = torch.softmax(energy, dim=-1)  # (batch, src_len)
        
        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch, hidden)
        
        return context, attn_weights


class Seq2SeqLSTMAttention(nn.Module):
    """LSTM encoder-decoder with bidirectional encoder and attention."""

    def __init__(self, config: Seq2SeqAttentionConfig) -> None:
        super().__init__()
        self.config = config
        
        # Shared embedding
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=config.pad_token_id,
        )
        self.dropout = nn.Dropout(config.dropout)
        
        # Bidirectional encoder
        self.encoder = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        
        # Projection layers to map bidirectional encoder to decoder size
        self.encoder_hidden_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.encoder_cell_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.encoder_output_proj = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
        # Attention mechanism
        self.attention = BahdanauAttention(config.hidden_dim)
        
        # Decoder with attention input
        self.decoder = nn.LSTM(
            input_size=config.embedding_dim + config.hidden_dim,  # embedding + context
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        
        # Output projection (decoder hidden + context)
        self.output_projection = nn.Linear(config.hidden_dim * 2, config.vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.output_projection.weight, -0.1, 0.1)
        nn.init.zeros_(self.output_projection.bias)

    def encode(self, source_ids: torch.Tensor, source_lengths: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Encode source sequence with bidirectional LSTM.
        
        Returns:
            hidden: (hidden, cell) tuple for decoder initialization
            encoder_outputs: All encoder hidden states for attention
        """
        embedded = self.dropout(self.embedding(source_ids))
        packed = pack_padded_sequence(embedded, source_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, cell) = self.encoder(packed)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        
        # Project bidirectional encoder outputs to decoder hidden size
        encoder_outputs = self.encoder_output_proj(encoder_outputs)
        
        # Combine forward and backward hidden states
        # hidden/cell: (num_layers * 2, batch, hidden_dim) -> (num_layers, batch, hidden_dim)
        hidden_fwd = hidden[0::2]  # Forward layers
        hidden_bwd = hidden[1::2]  # Backward layers
        cell_fwd = cell[0::2]
        cell_bwd = cell[1::2]
        
        # Concatenate and project
        hidden_combined = torch.cat([hidden_fwd, hidden_bwd], dim=-1)  # (num_layers, batch, hidden*2)
        cell_combined = torch.cat([cell_fwd, cell_bwd], dim=-1)
        
        hidden_proj = self.encoder_hidden_proj(hidden_combined)
        cell_proj = self.encoder_cell_proj(cell_combined)
        
        return (hidden_proj, cell_proj), encoder_outputs

    def decode(
        self,
        decoder_inputs: torch.Tensor,
        initial_state: Tuple[torch.Tensor, torch.Tensor],
        encoder_outputs: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode with attention over encoder outputs."""
        embedded = self.dropout(self.embedding(decoder_inputs))
        batch_size, seq_len, _ = embedded.size()
        
        decoder_hidden = initial_state
        outputs = []
        
        for t in range(seq_len):
            # Get current input
            curr_input = embedded[:, t:t+1, :]  # (batch, 1, embed_dim)
            
            # Compute attention over encoder outputs using current hidden state
            query = decoder_hidden[0][-1]  # Use last layer hidden state (batch, hidden)
            context, _ = self.attention(query, encoder_outputs, src_mask)
            
            # Concatenate context with input embedding
            decoder_input = torch.cat([curr_input, context.unsqueeze(1)], dim=-1)
            
            # Run one decoder step
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            
            # Concatenate decoder output with context for prediction
            combined = torch.cat([decoder_output.squeeze(1), context], dim=-1)
            outputs.append(combined)
        
        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden*2)
        logits = self.output_projection(self.dropout(outputs))
        
        return logits

    def forward(
        self,
        source_ids: torch.Tensor,
        source_lengths: torch.Tensor,
        decoder_inputs: torch.Tensor,
    ) -> torch.Tensor:
        hidden, encoder_outputs = self.encode(source_ids, source_lengths)
        
        # Create source mask for attention
        batch_size = source_ids.size(0)
        max_len = source_ids.size(1)
        src_mask = torch.arange(max_len, device=source_ids.device).unsqueeze(0) < source_lengths.unsqueeze(1)
        
        return self.decode(decoder_inputs, hidden, encoder_outputs, src_mask)

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
        """Greedy decoding with attention."""
        self.eval()
        device = device or source_ids.device

        hidden, encoder_outputs = self.encode(source_ids, source_lengths)
        hidden = (hidden[0].to(device), hidden[1].to(device))
        encoder_outputs = encoder_outputs.to(device)

        # Create source mask
        batch_size = source_ids.size(0)
        max_src_len = source_ids.size(1)
        src_mask = torch.arange(max_src_len, device=device).unsqueeze(0) < source_lengths.unsqueeze(1)

        inputs = torch.full((batch_size,), bos_token_id, dtype=torch.long, device=device)
        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            # Embed current input
            embedded = self.embedding(inputs.unsqueeze(1))  # (batch, 1, embed_dim)
            
            # Compute attention
            query = hidden[0][-1]  # (batch, hidden)
            context, _ = self.attention(query, encoder_outputs, src_mask)
            
            # Concatenate context with embedding
            decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
            
            # Decoder step
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            
            # Combine decoder output with context
            combined = torch.cat([decoder_output.squeeze(1), context], dim=-1)
            logits = self.output_projection(combined)
            
            next_tokens = torch.argmax(logits, dim=-1)
            outputs.append(next_tokens)

            finished = finished | (next_tokens == eos_token_id)
            if finished.all():
                break

            inputs = next_tokens

        if outputs:
            predictions = torch.stack(outputs, dim=1)
        else:
            predictions = torch.zeros((batch_size, 0), dtype=torch.long, device=device)
        return predictions

