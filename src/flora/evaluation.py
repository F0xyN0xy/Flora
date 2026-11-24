"""A small, transparent classical evaluation function.

Keep this function cheap: search calls it at a very large number of leaf nodes.
"""

from __future__ import annotations

import chess

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 335,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def evaluate(board: chess.Board) -> int:
    """Evaluate a position in centipawns from the side-to-move perspective."""
    if board.is_checkmate():
        return -100_000
    if board.is_game_over(claim_draw=True):
        return 0

    white_score = 0
    for square, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        value = PIECE_VALUES[piece.piece_type]
        file_distance = abs(3.5 - chess.square_file(square))
        rank = chess.square_rank(square)
        relative_rank = rank if piece.color == chess.WHITE else 7 - rank
        center_bonus = int(8 - 2 * (file_distance + abs(3.5 - rank)))
        positional = 0

        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            positional += center_bonus * 4
        elif piece.piece_type == chess.PAWN:
            positional += relative_rank * 7
            positional += int((3.5 - file_distance) * 3)
        elif piece.piece_type == chess.QUEEN:
            positional += center_bonus
        elif piece.piece_type == chess.KING and board.fullmove_number < 30:
            positional -= center_bonus * 3

        white_score += sign * (value + positional)

    # Deliberately no legal-move mobility count here. Generating legal moves
    # inside every leaf evaluation is expensive and search already generates
    # moves at interior nodes.
    return white_score if board.turn == chess.WHITE else -white_score
