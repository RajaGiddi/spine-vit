import torch
import torch.nn as nn
from torchvision.ops import roi_align


class Projection(nn.Module):
    def __init__(self, backbone_dim, embed_dim):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(backbone_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, roi_feats):
        x = self.pool(roi_feats).flatten(1)
        return self.act(self.norm(self.proj(x)))


class AnatomyTokenizer(nn.Module):
    def __init__(self, backbone_dim=384, embed_dim=256, roi_output_size=7,
                 spatial_scale=1 / 14, image_size=224):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.spatial_scale = spatial_scale
        self.image_size = image_size
        self.projection = Projection(backbone_dim, embed_dim)

    def forward(self, feature_map, boxes, level_indices=None, num_levels=None, images=None):
        roi_feats = roi_align(feature_map, boxes,
                              output_size=self.roi_output_size,
                              spatial_scale=self.spatial_scale,
                              aligned=True)
        return self.projection(roi_feats)


class UniformStripTokenizer(nn.Module):
    def __init__(self, backbone_dim=384, embed_dim=256, roi_output_size=7,
                 spatial_scale=1 / 14, image_size=224):
        super().__init__()
        self.roi_output_size = roi_output_size
        self.spatial_scale = spatial_scale
        self.image_size = image_size
        self.projection = Projection(backbone_dim, embed_dim)

    def make_strip_boxes(self, num_levels, device):
        size = float(self.image_size)
        rows = []
        for sample_index in range(len(num_levels)):
            count = num_levels[sample_index]
            if count <= 0:
                continue
            step = size / count
            for j in range(count):
                top = j * step
                bottom = (j + 1) * step
                rows.append([float(sample_index), 0.0, top, size, bottom])
        return torch.tensor(rows, dtype=torch.float32, device=device)

    def forward(self, feature_map, boxes, level_indices=None, num_levels=None, images=None):
        strip_boxes = self.make_strip_boxes(num_levels, feature_map.device)
        roi_feats = roi_align(feature_map, strip_boxes,
                              output_size=self.roi_output_size,
                              spatial_scale=self.spatial_scale,
                              aligned=True)
        return self.projection(roi_feats)


class PatchTokenizer(nn.Module):
    def __init__(self, backbone_dim=384, embed_dim=256, max_levels=12, num_heads=4,
                 image_size=224, spatial_scale=1 / 14):
        super().__init__()
        self.image_size = image_size
        self.spatial_scale = spatial_scale
        self.kv_proj = nn.Linear(backbone_dim, embed_dim)
        self.query_embed = nn.Embedding(max_levels, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, feature_map, boxes, level_indices=None, num_levels=None, images=None):
        patches = feature_map.flatten(2).transpose(1, 2)
        keys_values = self.kv_proj(patches)

        out_tokens = []
        offset = 0
        for sample_index in range(len(num_levels)):
            count = num_levels[sample_index]
            if count == 0:
                continue
            indices = level_indices[offset:offset + count]
            queries = self.query_embed(indices).unsqueeze(0)
            sample_kv = keys_values[sample_index:sample_index + 1]
            attended, _ = self.attn(queries, sample_kv, sample_kv)
            out_tokens.append(attended.squeeze(0))
            offset = offset + count

        tokens = torch.cat(out_tokens, dim=0)
        return self.act(self.norm(tokens))


class CASTCropTokenizer(nn.Module):
    def __init__(self, embed_dim=256, crop_size=112, freeze=True, image_size=224):
        super().__init__()
        import torchvision

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        resnet = torchvision.models.resnet18(weights=weights)
        layers = list(resnet.children())[:-1]
        self.encoder = nn.Sequential(*layers)

        self.freeze = freeze
        if freeze:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
            self.encoder.eval()

        self.crop_size = crop_size
        self.image_size = image_size
        self.proj = nn.Linear(512, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(self, feature_map, boxes, level_indices=None, num_levels=None, images=None):
        crops = roi_align(images, boxes, output_size=self.crop_size,
                          spatial_scale=1.0, aligned=True)

        if self.freeze:
            with torch.no_grad():
                feat = self.encoder(crops).flatten(1)
        else:
            feat = self.encoder(crops).flatten(1)

        return self.act(self.norm(self.proj(feat)))


def build_tokenizer(config):
    kind = config.get("tokenizer", "anatomy")
    backbone_dim = config.get("backbone_dim", 384)
    embed_dim = config.get("embed_dim", 256)
    image_size = config.get("image_size", 224)
    roi_output_size = config.get("roi_output_size", 7)
    spatial_scale = 1.0 / config.get("patch_size", 14)

    if kind == "anatomy":
        return AnatomyTokenizer(backbone_dim, embed_dim, roi_output_size,
                                spatial_scale, image_size)

    if kind == "strips":
        return UniformStripTokenizer(backbone_dim, embed_dim, roi_output_size,
                                     spatial_scale, image_size)

    if kind == "patches":
        return PatchTokenizer(backbone_dim, embed_dim, config.get("max_levels", 12),
                              config.get("encoder_heads", 4), image_size, spatial_scale)

    if kind == "cast_crop":
        return CASTCropTokenizer(embed_dim=embed_dim,
                                 crop_size=config.get("crop_size", 112),
                                 freeze=config.get("freeze_backbone", True),
                                 image_size=image_size)

    raise ValueError(f"Unknown tokenizer: {kind}")
