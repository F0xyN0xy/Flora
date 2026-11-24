"""Run a reproducible policy/value inference benchmark on CPU or CUDA."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter

import chess
import torch

from flora.encoding import encode_position
from flora.network import FloraNetwork


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    batch_size: int
    iterations: int
    elapsed_seconds: float
    positions_per_second: float
    peak_memory_mb: float | None


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve auto/cuda/cpu and fail clearly if CUDA was explicitly requested."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False. "
                "Check the NVIDIA driver and PyTorch CUDA installation."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def run_benchmark(
    *,
    device: str = "auto",
    batch_size: int = 32,
    warmup: int = 10,
    iterations: int = 50,
    channels: int = 64,
    residual_blocks: int = 4,
) -> BenchmarkResult:
    """Benchmark inference on one encoded chess position repeated in a batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must not be negative")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")

    target = resolve_device(device)
    model = FloraNetwork(channels=channels, residual_blocks=residual_blocks)
    model.to(target)
    model.eval()

    encoded = encode_position(chess.Board())
    batch = torch.from_numpy(encoded).unsqueeze(0).repeat(batch_size, 1, 1, 1)
    batch = batch.to(target)

    def synchronize() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)

    with torch.inference_mode():
        for _ in range(warmup):
            model(batch)
        synchronize()

        start = perf_counter()
        for _ in range(iterations):
            model(batch)
        synchronize()
        elapsed = perf_counter() - start

    total_positions = batch_size * iterations
    peak_memory = (
        torch.cuda.max_memory_allocated(target) / (1024 * 1024)
        if target.type == "cuda"
        else None
    )

    return BenchmarkResult(
        device=str(target),
        batch_size=batch_size,
        iterations=iterations,
        elapsed_seconds=elapsed,
        positions_per_second=total_positions / elapsed,
        peak_memory_mb=peak_memory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Flora neural-network inference.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--residual-blocks", type=int, default=4)
    args = parser.parse_args()

    result = run_benchmark(
        device=args.device,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
        channels=args.channels,
        residual_blocks=args.residual_blocks,
    )

    print(f"Flora inference benchmark")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA build: {torch.version.cuda or 'not available'}")
    print(f"Device: {result.device}")
    if result.device.startswith("cuda"):
        print(f"GPU: {torch.cuda.get_device_name(torch.device(result.device))}")
    print(f"Batch size: {result.batch_size}")
    print(f"Iterations: {result.iterations}")
    print(f"Elapsed: {result.elapsed_seconds:.3f} seconds")
    print(f"Throughput: {result.positions_per_second:,.1f} positions/second")
    if result.peak_memory_mb is not None:
        print(f"Peak allocated VRAM: {result.peak_memory_mb:.1f} MiB")


if __name__ == "__main__":
    main()
