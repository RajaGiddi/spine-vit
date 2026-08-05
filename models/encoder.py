from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn


class OrdinalPositionalEncoding(nn.Module):
    """Learned embedding over ordinal level positions."""

    def __init__(self, max_levels: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(max_levels, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, level_indices: torch.Tensor) -> torch.Tensor:
        return self.embed(level_indices)


class LearnedIdentityEncoding(nn.Module):
    def __init__(self, max_levels: int, embed_dim: int):
        super().__init__()
        self.embed = nn.Embedding(max_levels, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.10)

    def forward(self, level_indices: torch.Tensor) -> torch.Tensor:
        return self.embed(level_indices)


class NoPositionalEncoding(nn.Module):
    """No positional information - returns a zero tensor broadcastable to tokens."""

    def __init__(self, max_levels: int = 0, embed_dim: int = 0):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, level_indices: torch.Tensor) -> torch.Tensor:
        return torch.zeros(level_indices.shape[0], self.embed_dim, device=level_indices.device)


def build_pos_encoder(pos_encoding: str, max_levels: int, embed_dim: int) -> nn.Module:
    if pos_encoding == "ordinal":
        return OrdinalPositionalEncoding(max_levels, embed_dim)
    if pos_encoding == "learned":
        return LearnedIdentityEncoding(max_levels, embed_dim)
    if pos_encoding == "none":
        return NoPositionalEncoding(max_levels, embed_dim)
    raise ValueError(f"Unknown pos_encoding '{pos_encoding}' (expected ordinal|learned|none)")


class AnatomyEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_levels: int = 24,
        pos_encoding: str = "ordinal",
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_encoder = build_pos_encoder(pos_encoding, max_levels, embed_dim)
        self.type_embedding = nn.Embedding(2, embed_dim)
        nn.init.normal_(self.type_embedding.weight, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    @staticmethod
    def _pack(tokens: torch.Tensor, num_levels: List[int]) -> Tuple[torch.Tensor, torch.Tensor, list]:
        """Flat (N,D) -> padded (B, max_K, D) + key_padding_mask (True=pad) + slices."""
        b = len(num_levels)
        max_k = max(num_levels) if num_levels else 0
        d = tokens.shape[1]
        padded = tokens.new_zeros(b, max_k, d)
        mask = torch.ones(b, max_k, dtype=torch.bool, device=tokens.device)
        slices = []
        offset = 0
        for i, k in enumerate(num_levels):
            padded[i, :k] = tokens[offset : offset + k]
            mask[i, :k] = False
            slices.append((offset, k))
            offset += k
        return padded, mask, slices

    @staticmethod
    def _unpack(padded: torch.Tensor, slices, n_total: int) -> torch.Tensor:
        out = padded.new_zeros(n_total, padded.shape[-1])
        for i, (offset, k) in enumerate(slices):
            out[offset : offset + k] = padded[i, :k]
        return out

    def _add_context(self, tokens, level_indices, level_types):
        return tokens + self.pos_encoder(level_indices) + self.type_embedding(level_types)

    def forward(self, tokens, level_indices, level_types, num_levels):
        x = self._add_context(tokens, level_indices, level_types)
        padded, mask, slices = self._pack(x, num_levels)
        encoded = self.encoder(padded, src_key_padding_mask=mask)
        encoded = self.norm(encoded)
        return self._unpack(encoded, slices, tokens.shape[0])

    @torch.no_grad()
    def forward_with_attention(self, tokens, level_indices, level_types, num_levels):
        x = self._add_context(tokens, level_indices, level_types)
        padded, mask, slices = self._pack(x, num_levels)

        h = padded
        attn_maps = []
        for layer in self.encoder.layers:
            y = layer.norm1(h)
            attn_out, attn_w = layer.self_attn(
                y, y, y, key_padding_mask=mask, need_weights=True, average_attn_weights=True
            )
            h = h + layer.dropout1(attn_out)
            y2 = layer.norm2(h)
            ff = layer.linear2(layer.dropout(layer.activation(layer.linear1(y2))))
            h = h + layer.dropout2(ff)
            attn_maps.append(attn_w)
        h = self.norm(h)
        return self._unpack(h, slices, tokens.shape[0]), attn_maps
