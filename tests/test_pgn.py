import chess
import numpy as np

from flora.encoding import MOVE_VOCABULARY_SIZE, encode_position, move_to_index
from flora.pgn import pgn_to_examples


def test_pgn_conversion_creates_policy_and_side_to_move_values(tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(
        '[Event "Test"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 1-0\n',
        encoding="utf-8",
    )
    examples = pgn_to_examples(path)
    assert len(examples) == 3
    assert examples[0][0].shape == encode_position(chess.Board()).shape
    assert examples[0][1].shape == (MOVE_VOCABULARY_SIZE,)
    assert examples[0][1][move_to_index(chess.Move.from_uci("e2e4"))] == 1
    assert examples[0][2] == 1.0
    assert examples[1][2] == -1.0


def test_pgn_skips_unfinished_games(tmp_path):
    path = tmp_path / "unfinished.pgn"
    path.write_text('[Result "*"]\n\n1. e4 *\n', encoding="utf-8")
    assert pgn_to_examples(path) == []


def test_pgn_can_exclude_draws(tmp_path):
    path = tmp_path / "draw.pgn"
    path.write_text('[Result "1/2-1/2"]\n\n1. d4 d5 1/2-1/2\n', encoding="utf-8")
    assert pgn_to_examples(path, include_draws=False) == []
    assert len(pgn_to_examples(path, include_draws=True)) == 2


def test_pgn_respects_game_and_ply_limits(tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(
        '[Event "One"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n'
        '[Event "Two"]\n[Result "0-1"]\n\n1. d4 d5 0-1\n',
        encoding="utf-8",
    )
    examples = pgn_to_examples(path, max_games=1, max_plies=1)
    assert len(examples) == 1


def test_pgn_examples_retain_game_identity_across_positions_and_offsets(tmp_path):
    path = tmp_path / "games.pgn"
    path.write_text(
        '[Event "One"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n\n'
        '[Event "Two"]\n[Result "0-1"]\n\n1. d4 d5 0-1\n',
        encoding="utf-8",
    )
    examples = pgn_to_examples(path)
    assert [item[3] for item in examples] == [0, 0, 1, 1]

    from flora.pgn import iter_pgn_examples

    offset_examples = list(iter_pgn_examples(path, max_games=1, game_id_offset=12))
    assert offset_examples
    assert {item[3] for item in offset_examples} == {12}
