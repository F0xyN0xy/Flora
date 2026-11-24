import chess

from flora.evaluation import evaluate


def test_initial_position_is_near_balanced():
    assert abs(evaluate(chess.Board())) <= 20


def test_material_advantage_is_positive_for_side_to_move():
    board = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1")
    assert evaluate(board) > 0


def test_material_advantage_is_negative_for_opponent_to_move():
    board = chess.Board("4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1")
    assert evaluate(board) < 0


def test_checkmate_is_losing_for_side_to_move():
    board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    assert evaluate(board) == -100_000
