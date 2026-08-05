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

        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False
            self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()
        return self

    def extract(self, x):
        batch_size = x.shape[0]
        height = x.shape[2] // self.patch_size
        width = x.shape[3] // self.patch_size
        tokens = self.model.get_intermediate_layers(x, n=1)[0]
        return tokens.transpose(1, 2).reshape(batch_size, self.embed_dim, height, width)

    def forward(self, x):
        if self.freeze:
            with torch.no_grad():
                return self.extract(x)
        return self.extract(x)


class MockBackbone(nn.Module):
    def __init__(self, embed_dim=384, patch_size=14, freeze=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.spatial_scale = 1.0 / patch_size
        self.freeze = freeze
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
        feat = nn.functional.adaptive_avg_pool2d(feat, (height, width))
        if self.freeze:
            feat = feat.detach()
        return feat


def build_backbone(config):
    name = config.get("backbone", "dinov2_vits14")
    freeze = config.get("freeze_backbone", True)

    if name == "mock":
        return MockBackbone(embed_dim=config.get("backbone_dim", 384),
                            patch_size=config.get("patch_size", 14),
                            freeze=freeze)

    return DINOv2Backbone(model_name=name, freeze=freeze)
