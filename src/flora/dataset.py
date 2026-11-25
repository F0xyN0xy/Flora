"""Dataset storage helpers for Flora's training pipeline."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from flora.encoding import MOVE_VOCABULARY_SIZE, POSITION_PLANES


class FloraDataset(Dataset):
    """Load one NPZ shard. Keep shards small enough to fit comfortably in RAM."""

    def __init__(self, path: str | Path) -> None:
        with np.load(Path(path), allow_pickle=False) as data:
            self.positions = data["positions"].astype(np.float32, copy=False)
            self.policies = data["policies"].astype(np.float32, copy=False)
            self.values = data["values"].astype(np.float32, copy=False)
        if self.positions.ndim != 4 or self.positions.shape[1:] != (POSITION_PLANES, 8, 8):
            raise ValueError("Invalid positions array shape")
        if self.policies.shape != (len(self.positions), MOVE_VOCABULARY_SIZE):
            raise ValueError("Invalid policies array shape")
        if self.values.shape != (len(self.positions),):
            raise ValueError("Invalid values array shape")
        if not (len(self.positions) == len(self.policies) == len(self.values)):
            raise ValueError("Dataset arrays have different lengths")

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.positions[index]),
            torch.from_numpy(self.policies[index]),
            torch.tensor(self.values[index], dtype=torch.float32),
        )


def save_examples(path: str | Path, examples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
    """Write one compressed NPZ shard."""
    if not examples:
        raise ValueError("Cannot save an empty dataset")
    positions = np.stack([item[0] for item in examples]).astype(np.float16)
    policies = np.stack([item[1] for item in examples]).astype(np.float16)
    values = np.asarray([item[2] for item in examples], dtype=np.float32)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, positions=positions, policies=policies, values=values)


def write_chunked_dataset(
    output_dir: str | Path,
    examples: Iterable[tuple[np.ndarray, np.ndarray, float]],
    *,
    chunk_size: int = 512,
) -> tuple[int, int]:
    """Stream examples into bounded-size NPZ shards and write a manifest."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    existing = list(destination.glob("chunk-*.npz"))
    if existing:
        raise FileExistsError(
            f"{destination} already contains dataset chunks; choose an empty directory "
            "or move the existing chunks first"
        )

    buffer: list[tuple[np.ndarray, np.ndarray, float]] = []
    total = 0
    chunks = 0
    for example in examples:
        buffer.append(example)
        if len(buffer) >= chunk_size:
            save_examples(destination / f"chunk-{chunks:05d}.npz", buffer)
            total += len(buffer)
            chunks += 1
            buffer.clear()
            print(f"Saved chunk {chunks} ({total} positions)", flush=True)
    if buffer:
        save_examples(destination / f"chunk-{chunks:05d}.npz", buffer)
        total += len(buffer)
        chunks += 1
        print(f"Saved chunk {chunks} ({total} positions)", flush=True)
    if not total:
        raise ValueError("No training examples were produced")

    manifest = {
        "format": "flora-chunked-v1",
        "chunks": chunks,
        "positions": total,
        "chunk_size": chunk_size,
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
