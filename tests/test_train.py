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
        channels=8,
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
        channels=8,
        residual_blocks=1,
        checkpoint_path=checkpoint_path,
        validation_fraction=0.0,
    )

    resumed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert resumed["epoch"] == 2


def test_validation_split_keeps_each_game_in_only_one_split(tmp_path):
    from flora.dataset import write_chunked_dataset
    from flora.train import _split_chunks

    position = np.zeros((POSITION_PLANES, 8, 8), dtype=np.float32)
    policy = np.zeros(MOVE_VOCABULARY_SIZE, dtype=np.float32)
    policy[0] = 1.0
    game_ids = [100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103]
    examples = [(position, policy, 0.0, game_id) for game_id in game_ids]
    data_dir = tmp_path / "game-aware"
    write_chunked_dataset(data_dir, examples, chunk_size=4)

    chunks = __import__("flora.dataset", fromlist=["list_dataset_chunks"]).list_dataset_chunks(data_dir)
    training, validation, mode = _split_chunks(chunks, 0.25)

    def selected_game_ids(selections):
        selected = set()
        for path, indices in selections:
            with np.load(path, allow_pickle=False) as data:
                ids = data["game_ids"] if indices is None else data["game_ids"][indices]
                selected.update(ids.tolist())
        return selected

    training_ids = selected_game_ids(training)
    validation_ids = selected_game_ids(validation)
    assert mode == "game-aware"
    assert training_ids
    assert validation_ids
    assert training_ids.isdisjoint(validation_ids)
