"""Generate self-play games and policy/value training examples."""
from __future__ import annotations
import argparse
import random
from pathlib import Path
import chess
import numpy as np
import torch
from flora.dataset import save_examples
from flora.encoding import MOVE_VOCABULARY_SIZE, encode_position, move_to_index
from flora.mcts import MCTS
from flora.network import FloraNetwork

def play_game(network: torch.nn.Module, *, device: str = "cpu", simulations: int = 64, batch_size: int = 8, temperature_moves: int = 20, max_plies: int = 300, seed: int | None = None) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Play one self-play game; values are from each stored position's side-to-move view."""
    if simulations < 1 or batch_size < 1 or max_plies < 1:
        raise ValueError("simulations, batch_size and max_plies must be positive")
    rng = random.Random(seed)
    board = chess.Board()
    search = MCTS(network, device=device)
    recorded: list[tuple[np.ndarray, np.ndarray, bool]] = []
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        result = search.search(board, simulations=simulations, batch_size=batch_size)
        if result.best_move is None:
            break
        policy = np.zeros(MOVE_VOCABULARY_SIZE, dtype=np.float32)
        for move, visits in result.visit_counts.items():
            policy[move_to_index(move)] = visits
        total = float(policy.sum())
        if total:
            policy /= total
        recorded.append((encode_position(board), policy, board.turn))
        moves = list(result.visit_counts)
        weights = [result.visit_counts[move] for move in moves]
        move = rng.choices(moves, weights=weights, k=1)[0] if board.ply() < temperature_moves and sum(weights) else result.best_move
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    white_value = 0.0 if outcome is None or outcome.winner is None else (1.0 if outcome.winner == chess.WHITE else -1.0)
    return [(pos, policy, white_value if turn == chess.WHITE else -white_value) for pos, policy, turn in recorded]

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Flora self-play training data")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/selfplay.npz"))
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if min(args.games, args.simulations, args.batch_size, args.max_plies) < 1:
        parser.error("games, simulations, batch-size and max-plies must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    network = FloraNetwork()
    if args.checkpoint:
        network.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    network.to(device)
    examples = []
    for index in range(args.games):
        game = play_game(network, device=device, simulations=args.simulations, batch_size=args.batch_size, max_plies=args.max_plies, seed=args.seed + index)
        examples.extend(game)
        print(f"Game {index + 1}/{args.games}: {len(game)} positions")
    if not examples:
        parser.error("No positions generated")
    save_examples(args.output, examples)
    print(f"Saved {len(examples)} positions to {args.output}")

if __name__ == "__main__":
    main()
