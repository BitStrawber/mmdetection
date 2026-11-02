from __future__ import annotations

import hashlib
import json
import os
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


def existing_file(value: PathLike) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size <= 0:
        raise FileNotFoundError(f"Required file is missing or empty: {path}")
    return path


def ensure_empty_or_create(path: PathLike, overwrite: bool = False) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use --overwrite."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def read_json(path: PathLike) -> Any:
    with existing_file(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: PathLike, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(str(temporary), str(output))


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with existing_file(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"JSONL file has no rows: {path}")
    return rows


def write_jsonl(path: PathLike, rows: Iterable[Dict[str, Any]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("Refusing to write an empty JSONL manifest")
    os.replace(str(temporary), str(output))


def file_sha256(path: PathLike, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with existing_file(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def parse_csv(value: str | Sequence[str] | None) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(part).strip() for part in value if str(part).strip()]


def atomic_write_text(path: PathLike, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(str(temporary), str(output))
