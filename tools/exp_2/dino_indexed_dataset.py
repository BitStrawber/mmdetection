"""Compact ImageFolder-compatible dataset backed by nested path indexes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from torch.utils.data import Dataset


def _read_index(path: Path) -> list[str]:
    if not path.is_file():
        raise RuntimeError(f"index file does not exist: {path}")
    paths = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"index contains duplicate paths: {path}")
    return paths


class IndexedImageFolder(Dataset):
    """A torchvision ImageFolder replacement selecting a manifest path prefix."""

    def __init__(
        self,
        manifest_path: str | Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable] = None,
    ) -> None:
        from torchvision.datasets.folder import default_loader

        self.manifest_path = Path(manifest_path).expanduser().resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2 or payload.get("storage_mode") != "index_only":
            raise RuntimeError(f"unsupported indexed DINO manifest: {self.manifest_path}")
        self.root = Path(payload["source_root"]).expanduser().resolve()
        if not self.root.is_dir():
            raise RuntimeError(f"source ImageFolder root does not exist: {self.root}")

        base = _read_index(Path(payload["base_index_file"]))
        remaining = _read_index(Path(payload["remaining_index_file"]))
        base_count = int(payload["base_count"])
        additional_count = int(payload["additional_count"])
        expected_count = int(payload["image_count"])
        if len(base) != base_count or additional_count > len(remaining):
            raise RuntimeError(f"invalid index lengths in manifest: {self.manifest_path}")
        self.relative_paths = base + remaining[:additional_count]
        if len(self.relative_paths) != expected_count:
            raise RuntimeError(f"manifest image count mismatch: {self.manifest_path}")

        self.classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        if not self.classes:
            raise RuntimeError(f"ImageFolder root has no class directories: {self.root}")
        self.samples = []
        for relative in self.relative_paths:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or len(path.parts) < 2:
                raise RuntimeError(f"invalid ImageFolder-relative path: {relative}")
            class_name = path.parts[0]
            if class_name not in self.class_to_idx:
                raise RuntimeError(f"class directory absent from source root: {class_name}")
            self.samples.append((str(self.root / path), self.class_to_idx[class_name]))

        self.imgs = self.samples
        self.targets = [target for _, target in self.samples]
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader or default_loader

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
