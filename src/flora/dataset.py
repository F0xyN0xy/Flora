"""Compact storage and loading of self-play training examples."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from flora.encoding import MOVE_VOCABULARY_SIZE, POSITION_PLANES

class FloraDataset(Dataset):
    """Examples contain encoded positions, MCTS policy targets and game outcomes."""
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
        return (torch.from_numpy(self.positions[index]), torch.from_numpy(self.policies[index]), torch.tensor(self.values[index], dtype=torch.float32))

def save_examples(path: str | Path, examples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
    """Write examples as a compressed NPZ file."""
    if not examples:
        raise ValueError("Cannot save an empty dataset")
    positions = np.stack([item[0] for item in examples]).astype(np.float16)
    policies = np.stack([item[1] for item in examples]).astype(np.float16)
    values = np.asarray([item[2] for item in examples], dtype=np.float32)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, positions=positions, policies=policies, values=values)
