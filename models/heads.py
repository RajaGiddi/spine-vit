from __future__ import annotations

import torch
import torch.nn as nn


class CoralLayer(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        return self.fc(x) + self.bias


class GradingHeads(nn.Module):
    def __init__(self, embed_dim: int = 256, num_stenosis_classes: int = 3, num_pfirrmann_classes: int = 5,
                 dropout: float = 0.1, head_type: str = "ce"):
        super().__init__()
        self.head_type = head_type
        self.stenosis_head = self._mlp(embed_dim, num_stenosis_classes, dropout, head_type)
        self.pfirrmann_head = self._mlp(embed_dim, num_pfirrmann_classes, dropout, head_type)

    @staticmethod
    def _mlp(embed_dim, num_classes, dropout, head_type):
        final = CoralLayer(embed_dim, num_classes) if head_type == "coral" else nn.Linear(embed_dim, num_classes)
        return nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout), final)

    def forward(self, encoded_tokens: torch.Tensor, level_types: torch.Tensor, task: str = "stenosis"):
        disc_mask = level_types == 1
        disc_tokens = encoded_tokens[disc_mask]
        head = self.stenosis_head if task == "stenosis" else self.pfirrmann_head
        logits = head(disc_tokens)
        return logits, disc_mask
