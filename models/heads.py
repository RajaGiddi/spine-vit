import torch
import torch.nn as nn


class CoralLayer(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))

    def forward(self, x):
        return self.fc(x) + self.bias


def make_head(embed_dim, num_classes, dropout, head_type):
    if head_type == "coral":
        final = CoralLayer(embed_dim, num_classes)
    else:
        final = nn.Linear(embed_dim, num_classes)

    return nn.Sequential(
        nn.Linear(embed_dim, embed_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        final,
    )


class GradingHeads(nn.Module):
    def __init__(self, embed_dim=256, num_stenosis_classes=3, num_pfirrmann_classes=5,
                 dropout=0.1, head_type="ce"):
        super().__init__()
        self.head_type = head_type
        self.stenosis_head = make_head(embed_dim, num_stenosis_classes, dropout, head_type)
        self.pfirrmann_head = make_head(embed_dim, num_pfirrmann_classes, dropout, head_type)

    def forward(self, encoded_tokens, level_types, task="stenosis"):
        disc_mask = level_types == 1
        disc_tokens = encoded_tokens[disc_mask]

        if task == "stenosis":
            head = self.stenosis_head
        else:
            head = self.pfirrmann_head

        logits = head(disc_tokens)
        return logits, disc_mask
