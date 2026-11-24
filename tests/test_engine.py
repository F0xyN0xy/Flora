import chess

from flora.engine import Searcher


def test_search_returns_a_legal_move_and_does_not_mutate_board():
    board = chess.Board()
    before = board.fen()
    result = Searcher().search(board, max_depth=2)
    assert result.best_move in board.legal_moves
    assert board.fen() == before
    assert result.depth == 2
    assert result.nodes > 0
    assert result.pv[0] == result.best_move


def test_search_finds_mate_in_one():
    board = chess.Board("7k/6pp/5KQ1/8/8/8/8/8 w - - 0 1")
    result = Searcher().search(board, max_depth=3)
    assert result.best_move in board.legal_moves
    after = board.copy()
    after.push(result.best_move)
    assert after.is_checkmate()


def test_search_returns_no_move_for_terminal_position():
    board = chess.Board("7k/6Q1/5K2/8/8/8/8/8 b - - 0 1")
    result = Searcher().search(board, max_depth=3)
    assert result.best_move is None
    assert result.pv == ()


def test_search_validates_limits():
    import pytest
    with pytest.raises(ValueError):
        Searcher().search(chess.Board(), max_depth=0)
