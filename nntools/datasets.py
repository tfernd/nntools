from __future__ import annotations

from pathlib import Path
from PIL import Image

from torchvision.transforms import RandomCrop, ToTensor, Compose


class ImageDataset:
    paths: list[Path]

    def __init__(
        self, root: str, *, width: int, height: int, sufix: str = ".jpg",
    ):
        root = Path(root)

        self.paths = list(root.rglob(f"*.{sufix}"))

        self.transform = Compose([RandomCrop((height, width)), ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        img = Image.open(path).convert("RGB")

        return self.transform(img).permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
