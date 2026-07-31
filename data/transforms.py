"""Shared, box-aware augmentations for spine MRI.

The single hard rule here: every geometric transform applied to the image must be
applied *identically* to the bounding boxes. A vertical shift of the image by +dy
pixels shifts every box's y-coordinates by +dy; a horizontal flip mirrors the box
x-coordinates; a crop-and-resize rescales and offsets box coordinates.

Augmentations are training-only. Color jitter is intentionally omitted (meaningless
for grayscale MRI). Images are float32 in [0, 1]-ish range (z-scoring happens in the
dataset after augmentation).
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def _to_chw(image: np.ndarray):
    """Return (image_as_CHW, was_2d) so we can restore the caller's rank."""
    if image.ndim == 2:
        return image[None, :, :], True
    if image.ndim == 3:
        return image, False
    raise ValueError(f"Expected 2D (H,W) or 3D (C,H,W) image, got shape {image.shape}")


def _resize_chw(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize each channel of a (C, H, W) float32 array."""
    chans = []
    for c in range(image.shape[0]):
        pil = Image.fromarray(image[c].astype(np.float32), mode="F")
        pil = pil.resize((out_w, out_h), Image.BILINEAR)
        chans.append(np.asarray(pil, dtype=np.float32))
    return np.stack(chans, axis=0)


class SpineAugmentation:
    """Medical-image-appropriate augmentations with consistent box transforms.

    Args:
        image_size: output spatial size (H == W). Output image is guaranteed to be
            this size regardless of intermediate crops.
        p_hflip: probability of horizontal flip.
        vshift_frac: max vertical shift as a fraction of height (+/-).
        intensity_frac: max multiplicative intensity jitter (+/-).
        min_crop_area: minimum retained area fraction for random crop-and-resize.
        noise_std: std of additive Gaussian noise.
    """

    def __init__(
        self,
        image_size: int = 224,
        p_hflip: float = 0.5,
        vshift_frac: float = 0.10,
        intensity_frac: float = 0.05,
        min_crop_area: float = 0.85,
        noise_std: float = 0.01,
    ):
        self.image_size = image_size
        self.p_hflip = p_hflip
        self.vshift_frac = vshift_frac
        self.intensity_frac = intensity_frac
        self.min_crop_area = min_crop_area
        self.noise_std = noise_std

    def __call__(self, image: np.ndarray, boxes: np.ndarray):
        """Args:
            image: (H, W) or (C, H, W) float32 array.
            boxes: (K, 4) float array of [x1, y1, x2, y2] in image-pixel coords.
        Returns:
            (image, boxes) both augmented consistently. Image keeps the caller's rank.
        """
        img, was_2d = _to_chw(image)
        img = img.astype(np.float32).copy()
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        _, H, W = img.shape

        # 1. Random vertical shift (+/- vshift_frac): move image and boxes together.
        dy = int(round(np.random.uniform(-self.vshift_frac, self.vshift_frac) * H))
        if dy != 0:
            shifted = np.zeros_like(img)
            if dy > 0:
                shifted[:, dy:, :] = img[:, : H - dy, :]
            else:
                shifted[:, : H + dy, :] = img[:, -dy:, :]
            img = shifted
            boxes[:, [1, 3]] += dy

        # 2. Intensity jitter (+/- intensity_frac): multiplicative, no box change.
        img *= np.random.uniform(1.0 - self.intensity_frac, 1.0 + self.intensity_frac)

        # 3. Horizontal flip: mirror image and box x-coordinates.
        if np.random.rand() < self.p_hflip:
            img = img[:, :, ::-1].copy()
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = W - x2
            boxes[:, 2] = W - x1

        # 4. Random crop-and-resize: crop [min_crop_area, 1.0] of area, resize back.
        area_frac = np.random.uniform(self.min_crop_area, 1.0)
        side = float(np.sqrt(area_frac))
        crop_h = max(1, int(round(H * side)))
        crop_w = max(1, int(round(W * side)))
        top = np.random.randint(0, H - crop_h + 1)
        left = np.random.randint(0, W - crop_w + 1)
        img = img[:, top : top + crop_h, left : left + crop_w]
        # Adjust boxes into the crop frame, then rescale to the resized output.
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top
        sx = self.image_size / crop_w
        sy = self.image_size / crop_h
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
        img = _resize_chw(img, self.image_size, self.image_size)

        # 5. Additive Gaussian noise.
        if self.noise_std > 0:
            img = img + np.random.normal(0.0, self.noise_std, size=img.shape).astype(np.float32)

        # Clamp boxes to the (resized) image bounds and keep x1<=x2, y1<=y2.
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.image_size)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.image_size)
        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0])
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1])

        if was_2d:
            img = img[0]
        return img.astype(np.float32), boxes.astype(np.float32)
