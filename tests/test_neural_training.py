import numpy as np
import pytest
import torch
import chess

from flora.dataset import FloraDataset, save_examples
from flora.encoding import MOVE_VOCABULARY_SIZE, POSITION_PLANES
from flora.mcts import MCTS
from flora.network import FloraNetwork
from flora.selfplay import play_game


class RecordingNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []
    def forward(self, positions):
        self.batch_sizes.append(len(positions))
        return (
            torch.zeros(len(positions), MOVE_VOCABULARY_SIZE, device=positions.device),
            torch.zeros(len(positions), device=positions.device),
        )


def test_batched_mcts_returns_legal_move_and_all_visits():
    network = RecordingNetwork()
    board = chess.Board()
    result = MCTS(network).search(board, simulations=12, batch_size=4)
    assert result.best_move in board.legal_moves
    assert sum(result.visit_counts.values()) == 12
    assert max(network.batch_sizes) > 1


def test_batched_mcts_does_not_modify_board():
    board = chess.Board()
    original = board.fen()
    MCTS(RecordingNetwork()).search(board, simulations=7, batch_size=3)
    assert board.fen() == original


def test_dataset_round_trip(tmp_path):
    path = tmp_path / "samples.npz"
    position = np.zeros((POSITION_PLANES, 8, 8), dtype=np.float32)
    policy = np.zeros(MOVE_VOCABULARY_SIZE, dtype=np.float32)
    policy[123] = 1.0
    save_examples(path, [(position, policy, 1.0)])
    dataset = FloraDataset(path)
    x, p, value = dataset[0]
    assert len(dataset) == 1
    assert x.shape == (POSITION_PLANES, 8, 8)
    assert p[123].item() == pytest.approx(1.0)
    assert value.item() == pytest.approx(1.0)


def test_save_examples_rejects_empty_data(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_examples(tmp_path / "empty.npz", [])


def test_selfplay_respects_ply_limit_and_labels_positions():
    examples = play_game(
        RecordingNetwork(),
        simulations=2,
        batch_size=2,
        max_plies=2,
        seed=7,
    )
    assert len(examples) == 2
    for position, policy, value in examples:
        assert position.shape == (POSITION_PLANES, 8, 8)
        assert policy.shape == (MOVE_VOCABULARY_SIZE,)
        assert policy.sum() == pytest.approx(1.0)
        assert value == 0.0
