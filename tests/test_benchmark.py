import pytest
import torch

from flora.benchmark import resolve_device, run_benchmark


def test_auto_device_resolves_to_available_device():
    device = resolve_device("auto")
    assert device.type == ("cuda" if torch.cuda.is_available() else "cpu")


def test_explicit_cuda_requires_cuda_availability():
    if torch.cuda.is_available():
        assert resolve_device("cuda").type == "cuda"
    else:
        with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
            resolve_device("cuda")


def test_cpu_benchmark_returns_positive_throughput():
    result = run_benchmark(
        device="cpu",
        batch_size=2,
        warmup=0,
        iterations=1,
        channels=8,
        residual_blocks=1,
    )
    assert result.device == "cpu"
    assert result.batch_size == 2
    assert result.iterations == 1
    assert result.elapsed_seconds > 0
    assert result.positions_per_second > 0
    assert result.peak_memory_mb is None
