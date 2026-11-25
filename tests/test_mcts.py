import chess
import pytest
import torch

from flora.mcts import MCTS


class UniformNetwork(torch.nn.Module):
    def forward(self, positions):
        batch_size = positions.shape[0]
        logits = torch.zeros(batch_size, 64 * 64 * 5, device=positions.device)
        values = torch.zeros(batch_size, device=positions.device)
        return logits, values


def test_search_returns_a_legal_move_and_counts_all_simulations():
    board = chess.Board()
    result = MCTS(UniformNetwork()).search(board, simulations=12)

    assert result.best_move is not None
    assert result.best_move in board.legal_moves
    assert sum(result.visit_counts.values()) == 12
    assert set(result.visit_counts) == set(board.legal_moves)


def test_search_does_not_modify_the_input_board():
    board = chess.Board()
    original_fen = board.fen()

    MCTS(UniformNetwork()).search(board, simulations=5)

    assert board.fen() == original_fen


def test_checkmate_position_has_no_move_and_negative_value():
    board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")

    result = MCTS(UniformNetwork()).search(board, simulations=3)

    assert result.best_move is None
    assert result.visit_counts == {}
    assert result.root_value == -1.0


def test_stalemate_position_has_no_move_and_zero_value():
    board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")

    result = MCTS(UniformNetwork()).search(board, simulations=3)

    assert result.best_move is None
    assert result.root_value == 0.0


def test_search_requires_at_least_one_simulation():
    with pytest.raises(ValueError, match="simulations"):
        MCTS(UniformNetwork()).search(chess.Board(), simulations=0)


def test_c_puct_must_be_positive():
    with pytest.raises(ValueError, match="c_puct"):
        MCTS(UniformNetwork(), c_puct=0)
