"""Convert labelled PGN games into policy/value training examples."""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from flora.dataset import save_examples
from flora.encoding import MOVE_VOCABULARY_SIZE, encode_position, move_to_index


def pgn_to_examples(
    pgn_path: str | Path,
    *,
    max_games: int | None = None,
    max_plies: int | None = None,
    include_draws: bool = True,
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Read completed PGN games and create one-hot move policy targets.

    Values are from the perspective of the side to move in each position.
    Games with unknown or unsupported results are skipped.
    """
    if max_games is not None and max_games < 1:
        raise ValueError("max_games must be positive")
    if max_plies is not None and max_plies < 1:
        raise ValueError("max_plies must be positive")

    examples: list[tuple[np.ndarray, np.ndarray, float]] = []
    games_read = 0
    with Path(pgn_path).open("r", encoding="utf-8-sig", errors="replace") as handle:
        while max_games is None or games_read < max_games:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            games_read += 1

            result = game.headers.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue
            if result == "1/2-1/2" and not include_draws:
                continue

            white_value = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}[result]
            board = game.board()
            for ply, move in enumerate(game.mainline_moves()):
                if max_plies is not None and ply >= max_plies:
                    break
                if move not in board.legal_moves:
                    break
                position = encode_position(board)
                policy = np.zeros(MOVE_VOCABULARY_SIZE, dtype=np.float32)
                policy[move_to_index(move)] = 1.0
                value = white_value if board.turn == chess.WHITE else -white_value
                examples.append((position, policy, value))
                board.push(move)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PGN games into Flora training data")
    parser.add_argument("--input", type=Path, required=True, help="PGN file or directory of PGN files")
    parser.add_argument("--output", type=Path, default=Path("data/pgn_training.npz"))
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--exclude-draws", action="store_true")
    args = parser.parse_args()

    if args.input.is_dir():
        files = sorted(args.input.glob("*.pgn"))
        if not files:
            parser.error(f"No .pgn files found in {args.input}")
    elif args.input.is_file():
        files = [args.input]
    else:
        parser.error(f"Input does not exist: {args.input}")

    examples = []
    remaining = args.max_games
    for file_path in files:
        file_limit = remaining
        examples.extend(
            pgn_to_examples(
                file_path,
                max_games=file_limit,
                max_plies=args.max_plies,
                include_draws=not args.exclude_draws,
            )
        )
        if remaining is not None:
            # Count PGN records, including games skipped for unknown results.
            with file_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                records = 0
                while chess.pgn.read_game(handle) is not None:
                    records += 1
            remaining -= min(records, remaining)
            if remaining <= 0:
                break

    if not examples:
        parser.error("No labelled positions found in the supplied PGN data")
    save_examples(args.output, examples)
    print(f"Saved {len(examples)} positions from {len(files)} PGN file(s) to {args.output}")


if __name__ == "__main__":
    main()
