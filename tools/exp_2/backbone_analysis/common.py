from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Union

import numpy as np


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


PathLike = Union[str, Path]


def existing_file(value: PathLike) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f'Required file is missing or empty: {path}')
    return path


def ensure_empty_or_create(path: Path, overwrite: bool = False) -> Path:
    path = path.expanduser().resolve()
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f'Output directory is not empty: {path}. Pass --overwrite to reuse it.')
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: PathLike) -> Any:
    with existing_file(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def write_json(path: PathLike, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    os.replace(str(temporary), str(output))


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with existing_file(path).open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f'{path}:{line_number}: expected a JSON object')
            rows.append(value)
    if not rows:
        raise ValueError(f'Manifest has no records: {path}')
    return rows


def write_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    count = 0
    with temporary.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write('\n')
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError('Refusing to write an empty manifest')
    os.replace(str(temporary), str(output))


def parse_csv(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def sample_ids(rows: Sequence[Dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row['sample_index']) for row in rows], dtype=np.int64)


def validate_sample_order(rows: Sequence[Dict[str, Any]]) -> None:
    actual = [int(row['sample_index']) for row in rows]
    expected = list(range(len(rows)))
    if actual != expected:
        raise ValueError(
            'Manifest sample_index values must be contiguous and ordered from zero')
    image_ids = [int(row['image_id']) for row in rows]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError('Manifest contains duplicate image_id values')


def file_sha256(path: PathLike, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with existing_file(path).open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def finite_summary(value: np.ndarray) -> Dict[str, Any]:
    array = np.asarray(value)
    finite = np.isfinite(array)
    return {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'nan_count': int(np.isnan(array).sum()),
        'inf_count': int(np.isinf(array).sum()),
        'finite_count': int(finite.sum()),
        'min': float(array[finite].min()) if finite.any() else None,
        'max': float(array[finite].max()) if finite.any() else None,
        'mean': float(array[finite].mean()) if finite.any() else None,
    }
