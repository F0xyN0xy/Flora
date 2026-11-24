"""Stable encodings for chess positions and moves."""

from __future__ import annotations

import numpy as np
import chess

# Six piece types for White followed by six for Black, then:
# side to move, four castling-right planes, and en-passant square.
POSITION_PLANES = 18
BOARD_SIZE = 8

# A move is represented by (from square, to square, promotion piece).
# Promotion code 0 means no promotion; 1..4 map to N/B/R/Q.
PROMOTION_TO_CODE = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
CODE_TO_PROMOTION = {value: key for key, value in PROMOTION_TO_CODE.items()}
MOVE_VOCABULARY_SIZE = 64 * 64 * 5


def encode_position(board: chess.Board) -> np.ndarray:
    """Encode a board as a float32 tensor with shape (18, 8, 8).

    Piece planes use python-chess square indexing: rank 1 is row 0,
    file a is column 0. Metadata planes are filled across the board for
    side-to-move and castling rights; the en-passant plane marks one square.
    """
    planes = np.zeros((POSITION_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for square, piece in board.piece_map().items():
        color_offset = 0 if piece.color == chess.WHITE else 6
        plane = color_offset + piece.piece_type - 1
        row, col = chess.square_rank(square), chess.square_file(square)
        planes[plane, row, col] = 1.0

    if board.turn == chess.WHITE:
        planes[12, :, :] = 1.0

    castling_planes = {
        chess.WHITE: (13, 14),
        chess.BLACK: (15, 16),
    }
    for color, (kingside_plane, queenside_plane) in castling_planes.items():
        if board.has_kingside_castling_rights(color):
            planes[kingside_plane, :, :] = 1.0
        if board.has_queenside_castling_rights(color):
            planes[queenside_plane, :, :] = 1.0

    if board.ep_square is not None:
        row, col = chess.square_rank(board.ep_square), chess.square_file(board.ep_square)
        planes[17, row, col] = 1.0

    return planes


def move_to_index(move: chess.Move) -> int:
    """Map a chess move to an integer in [0, MOVE_VOCABULARY_SIZE)."""
    if move.promotion not in PROMOTION_TO_CODE:
        raise ValueError(f"Unsupported promotion piece: {move.promotion}")

    promotion_code = PROMOTION_TO_CODE[move.promotion]
    return (move.from_square * 64 + move.to_square) * 5 + promotion_code


def index_to_move(index: int) -> chess.Move:
    """Decode an integer produced by :func:`move_to_index`."""
    if not 0 <= index < MOVE_VOCABULARY_SIZE:
        raise ValueError(f"Move index out of range: {index}")

    promotion_code = index % 5
    from_to = index // 5
    to_square = from_to % 64
    from_square = from_to // 64
    return chess.Move(
        from_square,
        to_square,
        promotion=CODE_TO_PROMOTION[promotion_code],
    )
