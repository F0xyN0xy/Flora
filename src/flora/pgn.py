"""Stream labelled PGN games into chunked policy/value training data."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import numpy as np

from flora.dataset import write_chunked_dataset
from flora.encoding import MOVE_VOCABULARY_SIZE, encode_position, move_to_index


Example = tuple[np.ndarray, np.ndarray, float]


def iter_pgn_examples(
    pgn_path: str | Path,
    *,
    max_games: int | None = None,
    max_plies: int | None = None,
    include_draws: bool = True,
) -> Iterator[Example]:
    """Yield examples one position at a time; never retain all games in memory."""
    if max_games is not None and max_games < 1:
        raise ValueError("max_games must be positive")
    if max_plies is not None and max_plies < 1:
        raise ValueError("max_plies must be positive")

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
                yield position, policy, value
                board.push(move)


def pgn_to_examples(
    pgn_path: str | Path,
    *,
    max_games: int | None = None,
    max_plies: int | None = None,
    include_draws: bool = True,
) -> list[Example]:
    """Compatibility helper for tests and small PGN files; use the CLI for large files."""
    return list(iter_pgn_examples(
        pgn_path,
        max_games=max_games,
        max_plies=max_plies,
        include_draws=include_draws,
    ))


def iter_input_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        files = sorted(input_path.glob("*.pgn"))
        if not files:
            raise ValueError(f"No .pgn files found in {input_path}")
        return files
    if input_path.is_file():
        return [input_path]
    raise FileNotFoundError(f"Input does not exist: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream PGN games into chunked Flora training data")
    parser.add_argument("--input", type=Path, required=True, help="PGN file or directory of PGN files")
    parser.add_argument("--output", type=Path, default=Path("data/pgn_training"))
    parser.add_argument("--chunk-size", type=int, default=256, help="Positions per shard (default: 256)")
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-plies", type=int)
    parser.add_argument("--exclude-draws", action="store_true")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")

    try:
        files = iter_input_files(args.input)
        remaining = args.max_games

        def examples() -> Iterator[Example]:
            nonlocal remaining
            for file_path in files:
                if remaining is not None and remaining <= 0:
                    break
                for example in iter_pgn_examples(
                    file_path,
                    max_games=remaining,
                    max_plies=args.max_plies,
                    include_draws=not args.exclude_draws,
                ):
                    yield example
                if remaining is not None:
                    # Count all records, including unfinished games skipped for training.
                    with file_path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                        records = sum(1 for _ in iter(lambda: chess.pgn.read_game(handle), None))
                    remaining -= min(records, remaining)

        total, chunks = write_chunked_dataset(
            args.output, examples(), chunk_size=args.chunk_size
        )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(f"Saved {total} positions in {chunks} chunks to {args.output}")


if __name__ == "__main__":
    main()
