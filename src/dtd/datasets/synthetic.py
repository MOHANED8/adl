from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SyntheticShapesConfig:
    image_size: int = 32
    samples_per_split: int = 128
    seed: int = 123


class SyntheticShapesDataset(Dataset):
    """Deterministic synthetic RGB shapes dataset for fast local experiments."""

    classes = ["square", "cross", "circle"]

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        transform=None,
        config: SyntheticShapesConfig = SyntheticShapesConfig(),
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.config = config
        split_offset = {"train": 0, "val": 1, "test": 2}.get(split, 3)
        self.rng = np.random.default_rng(config.seed + split_offset)
        self.samples = self._generate_samples()

    def _generate_samples(self) -> list[tuple[Image.Image, int]]:
        samples: list[tuple[Image.Image, int]] = []
        for idx in range(self.config.samples_per_split):
            label = idx % len(self.classes)
            arr = self._render_shape(label)
            samples.append((Image.fromarray(arr, mode="RGB"), label))
        return samples

    def _render_shape(self, label: int) -> np.ndarray:
        size = self.config.image_size
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:] = np.array([20, 24, 32], dtype=np.uint8)
        jitter = int(self.rng.integers(-2, 3))
        center = size // 2 + jitter
        color = [
            np.array([220, 68, 55], dtype=np.uint8),
            np.array([59, 130, 246], dtype=np.uint8),
            np.array([34, 197, 94], dtype=np.uint8),
        ][label]
        radius = max(size // 5, 4)

        if label == 0:
            canvas[center - radius : center + radius, center - radius : center + radius] = color
        elif label == 1:
            canvas[center - radius : center + radius, center - 2 : center + 2] = color
            canvas[center - 2 : center + 2, center - radius : center + radius] = color
        else:
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - center) ** 2 + (yy - center) ** 2 <= radius**2
            canvas[mask] = color
        noise = self.rng.normal(loc=0.0, scale=6.0, size=canvas.shape)
        return np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image, label = self.samples[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
