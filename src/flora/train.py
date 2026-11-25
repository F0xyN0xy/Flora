"""Train Flora's policy/value network from legacy NPZ or chunked datasets."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from flora.dataset import FloraDataset, list_dataset_chunks
from flora.network import FloraNetwork

ChunkSelection = tuple[Path, np.ndarray | None]


def _split_chunks(
    chunk_paths: list[Path], validation_fraction: float
) -> tuple[list[ChunkSelection], list[ChunkSelection], str]:
    """Split by whole games when game IDs exist; otherwise use legacy shard split."""
    if validation_fraction <= 0:
        return [(path, None) for path in chunk_paths], [], "disabled"

    game_id_arrays: list[np.ndarray] = []
    has_game_ids = True
    for path in chunk_paths:
        with np.load(path, allow_pickle=False) as data:
            if "game_ids" not in data.files:
                has_game_ids = False
                break
            game_id_arrays.append(data["game_ids"].astype(np.int64, copy=False))

    if has_game_ids:
        threshold = int(validation_fraction * 10_000)
        if threshold < 1:
            threshold = 1
        training: list[ChunkSelection] = []
        validation: list[ChunkSelection] = []
        train_count = 0
        validation_count = 0
        for path, game_ids in zip(chunk_paths, game_id_arrays):
            # A deterministic modulo assignment keeps every position from a game
            # together, including games whose positions cross shard boundaries.
            validation_bucket = np.remainder(game_ids * 7_919 + 104_729, 10_000)
            validation_mask = validation_bucket < threshold
            validation_indices = np.flatnonzero(validation_mask)
            training_indices = np.flatnonzero(~validation_mask)
            if len(training_indices):
                training.append((path, training_indices))
                train_count += len(training_indices)
            if len(validation_indices):
                validation.append((path, validation_indices))
                validation_count += len(validation_indices)

        if train_count and validation_count:
            return training, validation, "game-aware"
        print(
            "Warning: game IDs are present, but this dataset is too small for the "
            "requested split; validation is disabled."
        )
        return [(path, None) for path in chunk_paths], [], "disabled"

    if len(chunk_paths) >= 2:
        split_at = max(1, int(len(chunk_paths) * (1.0 - validation_fraction)))
        split_at = min(split_at, len(chunk_paths) - 1)
        print(
            "Warning: dataset has no game IDs; using legacy shard-based validation. "
            "Re-import PGNs to enable game-aware validation."
        )
        return (
            [(path, None) for path in chunk_paths[:split_at]],
            [(path, None) for path in chunk_paths[split_at:]],
            "shard-based",
        )

    print("Validation disabled: game IDs or at least two dataset shards are required.")
    return [(path, None) for path in chunk_paths], [], "disabled"


def _load_checkpoint(
    network: FloraNetwork,
    checkpoint_path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[int, list[float], float]:
    """Load model weights and, when available, optimizer/training state."""
    checkpoint: Any = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict):
        network.load_state_dict(checkpoint)
        return 0, [], float("inf")

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    network.load_state_dict(state_dict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return (
        int(checkpoint.get("epoch", 0)),
        list(checkpoint.get("history", [])),
        float(checkpoint.get("best_validation_loss", float("inf"))),
    )


def _mean_loss(
    network: FloraNetwork,
    selections: list[ChunkSelection],
    batch_size: int,
    device: str,
    policy_loss_fn: nn.KLDivLoss,
    value_loss_fn: nn.MSELoss,
) -> float:
    """Evaluate selected positions without retaining all shards in memory."""
    network.eval()
    total_loss = 0.0
    samples = 0
    with torch.inference_mode():
        for chunk_path, indices in selections:
            dataset = FloraDataset(chunk_path, indices=indices)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            for positions, target_policy, target_value in loader:
                positions = positions.to(device)
                target_policy = target_policy.to(device)
                target_value = target_value.to(device)
                logits, values = network(positions)
                loss = (
                    policy_loss_fn(torch.log_softmax(logits, dim=1), target_policy)
                    + value_loss_fn(values, target_value)
                )
                count = len(positions)
                total_loss += float(loss) * count
                samples += count
            del loader, dataset
    return total_loss / samples if samples else float("nan")


def _save_checkpoint(
    output_path: str | Path,
    network: FloraNetwork,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: list[float],
    best_validation_loss: float,
) -> None:
    """Write a resumable checkpoint after each completed epoch."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "epoch": epoch,
            "model_state_dict": network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_validation_loss": best_validation_loss,
        },
        temporary,
    )
    temporary.replace(destination)


def train(
    dataset_path: str | Path,
    output_path: str | Path,
    *,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    channels: int = 64,
    residual_blocks: int = 4,
    checkpoint_path: str | Path | None = None,
    seed: int = 1,
    validation_fraction: float = 0.05,
) -> list[float]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    if not 0.0 <= validation_fraction < 0.5:
        raise ValueError("validation_fraction must be in the range [0, 0.5)")

    all_chunks = list_dataset_chunks(dataset_path)
    training_chunks, validation_chunks, split_mode = _split_chunks(
        all_chunks, validation_fraction
    )
    if not training_chunks:
        raise ValueError("Training split contains no positions")
    print(f"Dataset split: {split_mode}", flush=True)

    network = FloraNetwork(channels=channels, residual_blocks=residual_blocks)
    network.to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)

    start_epoch = 0
    history: list[float] = []
    best_validation_loss = float("inf")
    if checkpoint_path is not None:
        start_epoch, history, best_validation_loss = _load_checkpoint(
            network, checkpoint_path, optimizer
        )
        print(f"Resumed from epoch {start_epoch}: {checkpoint_path}")

    policy_loss_fn = nn.KLDivLoss(reduction="batchmean")
    value_loss_fn = nn.MSELoss()
    rng = random.Random(seed)

    for epoch_offset in range(epochs):
        epoch_number = start_epoch + epoch_offset + 1
        network.train()
        total_loss = 0.0
        samples = 0
        epoch_chunks = list(training_chunks)
        rng.shuffle(epoch_chunks)

        # Only one shard is loaded at a time, bounding host RAM usage.
        for chunk_path, indices in epoch_chunks:
            dataset = FloraDataset(chunk_path, indices=indices)
            if not len(dataset):
                continue
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for positions, target_policy, target_value in loader:
                positions = positions.to(device)
                target_policy = target_policy.to(device)
                target_value = target_value.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits, values = network(positions)
                policy_loss = policy_loss_fn(
                    torch.log_softmax(logits, dim=1), target_policy
                )
                value_loss = value_loss_fn(values, target_value)
                loss = policy_loss + value_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                count = len(positions)
                total_loss += float(loss.detach()) * count
                samples += count
            del loader, dataset

        if not samples:
            raise ValueError("Dataset contains no training examples")
        mean_loss = total_loss / samples
        history.append(mean_loss)
        message = (
            f"Epoch {epoch_number}/{start_epoch + epochs}: "
            f"loss={mean_loss:.5f} ({samples} positions)"
        )

        if validation_chunks:
            validation_loss = _mean_loss(
                network,
                validation_chunks,
                batch_size,
                device,
                policy_loss_fn,
                value_loss_fn,
            )
            message += f" val_loss={validation_loss:.5f}"
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss

        print(message, flush=True)
        _save_checkpoint(
            output_path,
            network,
            optimizer,
            epoch_number,
            history,
            best_validation_loss,
        )
        print(f"Checkpoint saved to {output_path}", flush=True)

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Flora from PGN or self-play data")
    parser.add_argument("--data", type=Path, required=True, help="NPZ file or chunked dataset directory")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/flora.pt"))
    parser.add_argument("--checkpoint", type=Path, help="Resume from a model/training checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of additional epochs to train")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = "cuda" if args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    ) else "cpu"
    train(
        args.data,
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=device,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
    )


if __name__ == "__main__":
    main()
