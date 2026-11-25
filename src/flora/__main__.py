"""Run a small environment and encoding smoke test with python -m flora."""

import chess

from flora.encoding import encode_position


def main() -> None:
    board = chess.Board()
    encoded = encode_position(board)

    print("Flora", __import__("flora").__version__)
    print()
    print(board)
    print()
    print(f"Position tensor: {encoded.shape} ({encoded.dtype})")
    print(f"Legal moves in the initial position: {board.legal_moves.count()}")


if __name__ == "__main__":
    main()
