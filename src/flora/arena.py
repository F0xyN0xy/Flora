"""Play Flora against an external UCI engine and save every game as PGN."""
from __future__ import annotations

import argparse
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import torch

from flora.mcts import MCTS
from flora.network import FloraNetwork


def load_network(checkpoint: Path, device: str) -> FloraNetwork:
    network = FloraNetwork()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    network.load_state_dict(state)
    network.to(device)
    network.eval()
    return network


def save_game(path: Path, board: chess.Board, *, number: int, flora_color: chess.Color, opponent: str) -> None:
    outcome = board.outcome(claim_draw=True)
    result = "*" if outcome is None else outcome.result()
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "Flora vs UCI Engine"
    game.headers["Site"] = "Local"
    game.headers["Round"] = str(number)
    game.headers["White"] = "Flora" if flora_color == chess.WHITE else opponent
    game.headers["Black"] = opponent if flora_color == chess.WHITE else "Flora"
    game.headers["Result"] = result
    if outcome is None:
        game.headers["Termination"] = "Ply limit reached; game is unfinished"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        game.accept(chess.pgn.FileExporter(handle))
        handle.write("\n\n")


def play_match(
    network: FloraNetwork,
    engine: chess.engine.SimpleEngine,
    *,
    flora_color: chess.Color,
    simulations: int,
    batch_size: int,
    opponent_time: float,
    max_plies: int,
) -> chess.Board:
    board = chess.Board()
    search = MCTS(network, device=next(network.parameters()).device)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        if board.turn == flora_color:
            result = search.search(board, simulations=simulations, batch_size=batch_size)
            if result.best_move is None:
                break
            board.push(result.best_move)
        else:
            response = engine.play(board, chess.engine.Limit(time=opponent_time))
            if response.move is None:
                break
            board.push(response.move)
    return board


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Flora against an installed UCI chess engine")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained Flora .pt checkpoint")
    parser.add_argument("--opponent", type=Path, required=True, help="Path to opponent UCI engine executable")
    parser.add_argument("--pgn-output", type=Path, default=Path("data/arena_games.pgn"))
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--opponent-time", type=float, default=0.25, help="Opponent seconds per move")
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--flora-color", choices=("white", "black", "alternate"), default="alternate")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    if not args.opponent.is_file():
        parser.error(f"Opponent executable not found: {args.opponent}")
    if min(args.games, args.simulations, args.batch_size, args.max_plies) < 1 or args.opponent_time <= 0:
        parser.error("games, simulations, batch-size, max-plies and opponent-time must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    network = load_network(args.checkpoint, device)

    engine = chess.engine.SimpleEngine.popen_uci(str(args.opponent))
    try:
        print(f"Opponent: {engine.id.get('name', args.opponent.name)}")
        for index in range(args.games):
            if args.flora_color == "alternate":
                flora_color = chess.WHITE if index % 2 == 0 else chess.BLACK
            else:
                flora_color = args.flora_color == "white"
            board = play_match(
                network,
                engine,
                flora_color=flora_color,
                simulations=args.simulations,
                batch_size=args.batch_size,
                opponent_time=args.opponent_time,
                max_plies=args.max_plies,
            )
            save_game(
                args.pgn_output,
                board,
                number=index + 1,
                flora_color=flora_color,
                opponent=engine.id.get("name", args.opponent.stem),
            )
            outcome = board.outcome(claim_draw=True)
            result = "*" if outcome is None else outcome.result()
            print(f"Game {index + 1}/{args.games}: {result}; {board.ply()} plies; saved to {args.pgn_output}")
    finally:
        engine.quit()


if __name__ == "__main__":
    main()
