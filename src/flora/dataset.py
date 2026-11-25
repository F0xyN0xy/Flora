"""Dataset storage helpers for Flora's training pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from flora.encoding import MOVE_VOCABULARY_SIZE, POSITION_PLANES

Example = tuple[np.ndarray, np.ndarray, float]


class FloraDataset(Dataset):
    """Load one NPZ shard, optionally exposing only selected position indices."""

    def __init__(
        self,
        path: str | Path,
        indices: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        with np.load(Path(path), allow_pickle=False) as data:
            self.positions = data["positions"].astype(np.float32, copy=False)
            self.policies = data["policies"].astype(np.float32, copy=False)
            self.values = data["values"].astype(np.float32, copy=False)
            self.game_ids = (
                data["game_ids"].astype(np.int64, copy=False)
                if "game_ids" in data.files else None
            )
        if self.positions.ndim != 4 or self.positions.shape[1:] != (POSITION_PLANES, 8, 8):
            raise ValueError("Invalid positions array shape")
        if self.policies.shape != (len(self.positions), MOVE_VOCABULARY_SIZE):
            raise ValueError("Invalid policies array shape")
        if self.values.shape != (len(self.positions),):
            raise ValueError("Invalid values array shape")
        if not (len(self.positions) == len(self.policies) == len(self.values)):
            raise ValueError("Dataset arrays have different lengths")
        if self.game_ids is not None and self.game_ids.shape != (len(self.positions),):
            raise ValueError("Invalid game_ids array shape")

        self.indices = (
            np.arange(len(self.values), dtype=np.int64)
            if indices is None else np.asarray(indices, dtype=np.int64)
        )
        if self.indices.ndim != 1 or np.any(self.indices < 0) or np.any(self.indices >= len(self.values)):
            raise ValueError("Dataset indices are out of range")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        source_index = int(self.indices[index])
        return (
            torch.from_numpy(self.positions[source_index]),
            torch.from_numpy(self.policies[source_index]),
            torch.tensor(self.values[source_index], dtype=torch.float32),
        )


def save_examples(
    path: str | Path,
    examples: list[tuple[np.ndarray, np.ndarray, float]],
    *,
    game_ids: Sequence[int] | np.ndarray | None = None,
) -> None:
    """Write one compressed NPZ shard, optionally retaining per-position game IDs."""
    if not examples:
        raise ValueError("Cannot save an empty dataset")
    positions = np.stack([item[0] for item in examples]).astype(np.float16)
    policies = np.stack([item[1] for item in examples]).astype(np.float16)
    values = np.asarray([item[2] for item in examples], dtype=np.float32)
    arrays = {"positions": positions, "policies": policies, "values": values}
    if game_ids is not None:
        ids = np.asarray(game_ids, dtype=np.int64)
        if ids.shape != (len(examples),):
            raise ValueError("game_ids must contain one ID per example")
        arrays["game_ids"] = ids
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)


def write_chunked_dataset(
    output_dir: str | Path,
    examples: Iterable[tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, int]],
    *,
    chunk_size: int = 512,
) -> tuple[int, int]:
    """Stream examples into bounded-size NPZ shards and write a manifest.

    Four-element examples carry a game ID. Three-element examples remain
    supported for self-play and older callers, but cannot be game-split later.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if list(destination.glob("chunk-*.npz")):
        raise FileExistsError(
            f"{destination} already contains dataset chunks; choose an empty directory "
            "or move the existing chunks first"
        )

    buffer: list[tuple[np.ndarray, np.ndarray, float] | tuple[np.ndarray, np.ndarray, float, int]] = []
    total = 0
    chunks = 0
    has_game_ids: bool | None = None

    def flush() -> None:
        nonlocal total, chunks
        if not buffer:
            return
        examples_3 = [(item[0], item[1], item[2]) for item in buffer]
        ids = [item[3] for item in buffer] if has_game_ids else None
        save_examples(destination / f"chunk-{chunks:05d}.npz", examples_3, game_ids=ids)
        total += len(buffer)
        chunks += 1
        buffer.clear()
        print(f"Saved chunk {chunks} ({total} positions)", flush=True)

    for example in examples:
        carries_id = len(example) == 4
        if has_game_ids is None:
            has_game_ids = carries_id
        elif carries_id != has_game_ids:
            raise ValueError("All examples must either include game IDs or omit them")
        buffer.append(example)
        if len(buffer) >= chunk_size:
            flush()

    flush()
    if not total:
        raise ValueError("No training examples were produced")

    manifest = {
        "format": "flora-chunked-v2" if has_game_ids else "flora-chunked-v1",
        "chunks": chunks,
        "positions": total,
        "chunk_size": chunk_size,
        "game_ids": bool(has_game_ids),
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return total, chunks


def list_dataset_chunks(path: str | Path) -> list[Path]:
    """Return NPZ shards for either a chunk directory or a legacy NPZ file."""
    source = Path(path)
    if source.is_file():
        return [source]
    if source.is_dir():
        chunks = sorted(source.glob("chunk-*.npz"))
        if chunks:
            return chunks
    raise FileNotFoundError(f"No Flora dataset or chunk files found at {source}")
