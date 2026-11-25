import chess
import chess.pgn

from flora.selfplay import append_pgn


def test_append_pgn_saves_a_parseable_game(tmp_path):
    board = chess.Board()
    board.push_uci("e2e4")
    board.push_uci("e7e5")
    path = tmp_path / "games" / "selfplay.pgn"

    append_pgn(path, board, game_number=1)

    with path.open("r", encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    assert game is not None
    assert game.headers["Event"] == "Flora Self-Play"
    assert game.headers["Result"] == "*"
    assert [move.uci() for move in game.mainline_moves()] == ["e2e4", "e7e5"]
