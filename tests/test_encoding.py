import chess
import numpy as np
import pytest

from flora.encoding import (
    MOVE_VOCABULARY_SIZE,
    POSITION_PLANES,
    encode_position,
    index_to_move,
    move_to_index,
)


def test_initial_position_shape_and_dtype():
    encoded = encode_position(chess.Board())
    assert encoded.shape == (POSITION_PLANES, 8, 8)
    assert encoded.dtype == np.float32


def test_initial_position_piece_counts():
    encoded = encode_position(chess.Board())
    assert int(encoded[:6].sum()) == 16
    assert int(encoded[6:12].sum()) == 16


def test_side_to_move_plane():
    board = chess.Board()
    assert np.all(encode_position(board)[12] == 1)

    board.push_uci("e2e4")
    assert np.all(encode_position(board)[12] == 0)


def test_castling_rights_are_encoded():
    board = chess.Board()
    encoded = encode_position(board)
    assert np.all(encoded[13] == 1)
    assert np.all(encoded[14] == 1)
    assert np.all(encoded[15] == 1)
    assert np.all(encoded[16] == 1)


@pytest.mark.parametrize(
    "uci",
    ["e2e4", "g1f3", "e7e8q", "a7a8n", "b7a8r", "c7c8b"],
)
def test_move_round_trip(uci):
    move = chess.Move.from_uci(uci)
    assert index_to_move(move_to_index(move)) == move


def test_move_index_bounds():
    with pytest.raises(ValueError):
        index_to_move(-1)
    with pytest.raises(ValueError):
        index_to_move(MOVE_VOCABULARY_SIZE)
