import torch
import torch.nn as nn


class OrdinalPositionalEncoding(nn.Module):
    def __init__(self, max_levels, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(max_levels, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, level_indices):
        return self.embed(level_indices)


class LearnedIdentityEncoding(nn.Module):
    def __init__(self, max_levels, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(max_levels, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.10)

    def forward(self, level_indices):
        return self.embed(level_indices)


class NoPositionalEncoding(nn.Module):
    def __init__(self, max_levels=0, embed_dim=0):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, level_indices):
        return torch.zeros(level_indices.shape[0], self.embed_dim, device=level_indices.device)


def build_pos_encoder(pos_encoding, max_levels, embed_dim):
    if pos_encoding == "ordinal":
        return OrdinalPositionalEncoding(max_levels, embed_dim)
    if pos_encoding == "learned":
        return LearnedIdentityEncoding(max_levels, embed_dim)
    if pos_encoding == "none":
        return NoPositionalEncoding(max_levels, embed_dim)
    raise ValueError(f"Unknown pos_encoding: {pos_encoding}")


class AnatomyEncoder(nn.Module):
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, dropout=0.1,
                 max_levels=24, pos_encoding="ordinal"):
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
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

    def pack(self, tokens, num_levels):
        batch_size = len(num_levels)
        if num_levels:
            max_count = max(num_levels)
        else:
            max_count = 0
        token_dim = tokens.shape[1]

        padded = tokens.new_zeros(batch_size, max_count, token_dim)
        mask = torch.ones(batch_size, max_count, dtype=torch.bool, device=tokens.device)
        slices = []

        offset = 0
        for i in range(batch_size):
            count = num_levels[i]
            padded[i, :count] = tokens[offset:offset + count]
            mask[i, :count] = False
            slices.append((offset, count))
            offset = offset + count

        return padded, mask, slices

    def unpack(self, padded, slices, n_total):
        out = padded.new_zeros(n_total, padded.shape[-1])
        for i in range(len(slices)):
            offset, count = slices[i]
            out[offset:offset + count] = padded[i, :count]
        return out

    def add_context(self, tokens, level_indices, level_types):
        return tokens + self.pos_encoder(level_indices) + self.type_embedding(level_types)

    def forward(self, tokens, level_indices, level_types, num_levels):
        tokens_with_context = self.add_context(tokens, level_indices, level_types)
        padded, mask, slices = self.pack(tokens_with_context, num_levels)
        encoded = self.encoder(padded, src_key_padding_mask=mask)
        encoded = self.norm(encoded)
        return self.unpack(encoded, slices, tokens.shape[0])

    @torch.no_grad()
    def forward_with_attention(self, tokens, level_indices, level_types, num_levels):
        tokens_with_context = self.add_context(tokens, level_indices, level_types)
        padded, mask, slices = self.pack(tokens_with_context, num_levels)

        hidden = padded
        attention_maps = []
        for layer in self.encoder.layers:
            normed = layer.norm1(hidden)
            attn_out, attn_weights = layer.self_attn(normed, normed, normed, key_padding_mask=mask,
                                                     need_weights=True,
                                                     average_attn_weights=True)
            hidden = hidden + layer.dropout1(attn_out)

            normed_again = layer.norm2(hidden)
            feed_forward = layer.linear2(layer.dropout(layer.activation(layer.linear1(normed_again))))
            hidden = hidden + layer.dropout2(feed_forward)

            attention_maps.append(attn_weights)

        hidden = self.norm(hidden)
        return self.unpack(hidden, slices, tokens.shape[0]), attention_maps
