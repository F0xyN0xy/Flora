import numpy as np
import torch

from flora.dataset import save_examples
from flora.encoding import MOVE_VOCABULARY_SIZE, POSITION_PLANES
from flora.train import train


def _write_tiny_dataset(path):
    position = np.zeros((POSITION_PLANES, 8, 8), dtype=np.float32)
    policy = np.zeros(MOVE_VOCABULARY_SIZE, dtype=np.float32)
    policy[0] = 1.0
    save_examples(path, [(position, policy, 0.0)])


def test_training_saves_resumable_checkpoint(tmp_path):
    dataset_path = tmp_path / "tiny.npz"
    checkpoint_path = tmp_path / "flora.pt"
    _write_tiny_dataset(dataset_path)

    history = train(
        dataset_path,
        checkpoint_path,
        epochs=1,
        batch_size=1,
        device="cpu",
        channels=4,
        residual_blocks=1,
        validation_fraction=0.0,
    )

    assert len(history) == 1
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint

    train(
        dataset_path,
        checkpoint_path,
        epochs=1,
        batch_size=1,
        device="cpu",
        channels=4,
        residual_blocks=1,
        checkpoint_path=checkpoint_path,
        validation_fraction=0.0,
    )

    resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert resumed["epoch"] == 2
