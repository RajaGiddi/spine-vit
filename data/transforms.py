import numpy as np
from PIL import Image


def to_chw(image):
    if image.ndim == 2:
        return image[None, :, :], True
    if image.ndim == 3:
        return image, False
    raise ValueError(f"Expected a (H,W) or (C,H,W) image, got shape {image.shape}")


def resize_channels(image, out_h, out_w):
    channels = []
    for channel in range(image.shape[0]):
        pil = Image.fromarray(image[channel].astype(np.float32), mode="F")
        pil = pil.resize((out_w, out_h), Image.BILINEAR)
        channels.append(np.asarray(pil, dtype=np.float32))
    return np.stack(channels, axis=0)


class SpineAugmentation:
    def __init__(self, image_size=224, p_hflip=0.5, vshift_frac=0.10,
                 intensity_frac=0.05, min_crop_area=0.85, noise_std=0.01):
        self.image_size = image_size
        self.p_hflip = p_hflip
        self.vshift_frac = vshift_frac
        self.intensity_frac = intensity_frac
        self.min_crop_area = min_crop_area
        self.noise_std = noise_std

    def vertical_shift(self, image, boxes, height):
        shift = int(round(np.random.uniform(-self.vshift_frac, self.vshift_frac) * height))
        if shift == 0:
            return image, boxes

        shifted = np.zeros_like(image)
        if shift > 0:
            shifted[:, shift:, :] = image[:, :height - shift, :]
        else:
            shifted[:, :height + shift, :] = image[:, -shift:, :]

        boxes[:, [1, 3]] += shift
        return shifted, boxes

    def horizontal_flip(self, image, boxes, width):
        image = image[:, :, ::-1].copy()
        left = boxes[:, 0].copy()
        right = boxes[:, 2].copy()
        boxes[:, 0] = width - right
        boxes[:, 2] = width - left
        return image, boxes

    def crop_and_resize(self, image, boxes, height, width):
        area_frac = np.random.uniform(self.min_crop_area, 1.0)
        side = float(np.sqrt(area_frac))
        crop_h = max(1, int(round(height * side)))
        crop_w = max(1, int(round(width * side)))

        top = np.random.randint(0, height - crop_h + 1)
        left = np.random.randint(0, width - crop_w + 1)

        image = image[:, top:top + crop_h, left:left + crop_w]
        boxes[:, [0, 2]] -= left
        boxes[:, [1, 3]] -= top

        boxes[:, [0, 2]] *= self.image_size / crop_w
        boxes[:, [1, 3]] *= self.image_size / crop_h

        image = resize_channels(image, self.image_size, self.image_size)
        return image, boxes

    def __call__(self, image, boxes):
        image, was_2d = to_chw(image)
        image = image.astype(np.float32).copy()
        boxes = np.asarray(boxes, dtype=np.float32).copy()
        height = image.shape[1]
        width = image.shape[2]

        image, boxes = self.vertical_shift(image, boxes, height)

        image = image * np.random.uniform(1.0 - self.intensity_frac,
                                          1.0 + self.intensity_frac)

        if np.random.rand() < self.p_hflip:
            image, boxes = self.horizontal_flip(image, boxes, width)

        image, boxes = self.crop_and_resize(image, boxes, height, width)

        if self.noise_std > 0:
            noise = np.random.normal(0.0, self.noise_std, size=image.shape)
            image = image + noise.astype(np.float32)

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.image_size)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.image_size)
        boxes[:, 2] = np.maximum(boxes[:, 2], boxes[:, 0])
        boxes[:, 3] = np.maximum(boxes[:, 3], boxes[:, 1])

        if was_2d:
            image = image[0]

        return image.astype(np.float32), boxes.astype(np.float32)
