import torch
import torch.nn as nn


class DINOv2Backbone(nn.Module):
    def __init__(self, model_name="dinov2_vits14", freeze=True):
        super().__init__()
        self.model_name = model_name
        self.freeze = freeze
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True)
        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.spatial_scale = 1.0 / self.patch_size

        # We only train the small head on top, the backbone stays as it came
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        # model.train() would flip the backbone back on, so put it back in eval
        if self.freeze:
            self.model.eval()
        return self

    def extract(self, x):
        batch_size = x.shape[0]
        height = x.shape[2] // self.patch_size
        width = x.shape[3] // self.patch_size

        # Dinov2 hands back a flat list of patch tokens, we want it as a picture again
        tokens = self.model.get_intermediate_layers(x, n=1)[0]
        moved = tokens.transpose(1, 2)
        return moved.reshape(batch_size, self.embed_dim, height, width)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                return self.extract(x)
        else:
            return self.extract(x)


class MockBackbone(nn.Module):
    def __init__(self, embed_dim=384, patch_size=14, freeze=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.spatial_scale = 1.0 / patch_size
        self.freeze = freeze

        # Random cnn with the same output shape as dinov2, so the tests run offline
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, embed_dim, 3, stride=2, padding=1),
            nn.GELU(),
        )

        if freeze:
            for parameter in self.parameters():
                parameter.requires_grad = False
            self.stem.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.stem.eval()
        return self

    def forward(self, x):
        height = x.shape[2] // self.patch_size
        width = x.shape[3] // self.patch_size

        feat = self.stem(x)
        # Pool down to whatever grid dinov2 would have produced
        feat = nn.functional.adaptive_avg_pool2d(feat, (height, width))

        if self.freeze:
            feat = feat.detach()
        return feat


def build_backbone(config):
    name = config.get("backbone", "dinov2_vits14")
    freeze = config.get("freeze_backbone", True)

    if name == "mock":
        embed_dim = config.get("backbone_dim", 384)
        patch_size = config.get("patch_size", 14)
        return MockBackbone(embed_dim=embed_dim, patch_size=patch_size, freeze=freeze)
    else:
        return DINOv2Backbone(model_name=name, freeze=freeze)
