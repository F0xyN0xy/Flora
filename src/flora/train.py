"""Train Flora's policy/value network from saved self-play examples."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from flora.dataset import FloraDataset
from flora.network import FloraNetwork

def train(dataset_path: str | Path, output_path: str | Path, *, epochs: int = 5, batch_size: int = 32, learning_rate: float = 1e-3, device: str = "cpu", channels: int = 64, residual_blocks: int = 4) -> list[float]:
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size and learning_rate must be positive")
    dataset = FloraDataset(dataset_path)
    if not len(dataset):
        raise ValueError("Dataset is empty")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    network = FloraNetwork(channels=channels, residual_blocks=residual_blocks).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate)
    policy_loss_fn, value_loss_fn = nn.KLDivLoss(reduction="batchmean"), nn.MSELoss()
    history = []
    for epoch in range(epochs):
        network.train()
        total_loss = 0.0
        samples = 0
        for positions, target_policy, target_value in loader:
            positions, target_policy, target_value = positions.to(device), target_policy.to(device), target_value.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, values = network(positions)
            policy_loss = policy_loss_fn(torch.log_softmax(logits, dim=1), target_policy)
            value_loss = value_loss_fn(values, target_value)
            loss = policy_loss + value_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(positions)
            samples += len(positions)
        mean_loss = total_loss / samples
        history.append(mean_loss)
        print(f"Epoch {epoch + 1}/{epochs}: loss={mean_loss:.5f}")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(network.state_dict(), destination)
    print(f"Saved checkpoint to {destination}")
    return history

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Flora from self-play data")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/flora.pt"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    train(args.data, args.output, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, device=device)

if __name__ == "__main__":
    main()
