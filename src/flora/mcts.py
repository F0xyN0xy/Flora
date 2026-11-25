"""Batched policy/value-guided Monte Carlo Tree Search."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from flora.encoding import encode_position, move_to_index


@dataclass
class _Node:
    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    virtual_visits: int = 0
    pending: bool = False
    expanded: bool = False
    children: dict[chess.Move, "_Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    best_move: chess.Move | None
    visit_counts: dict[chess.Move, int]
    root_value: float


class MCTS:
    """PUCT search with batched neural inference for selected leaf positions."""

    def __init__(self, network: torch.nn.Module, *, device: str | torch.device = "cpu", c_puct: float = 1.5) -> None:
        if c_puct <= 0:
            raise ValueError("c_puct must be greater than zero")
        self.network = network
        self.device = torch.device(device)
        self.c_puct = c_puct
        self.network.to(self.device)

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            raise ValueError("Position is not terminal")
        if outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0

    def _evaluate_batch(self, boards: list[chess.Board]) -> tuple[list[np.ndarray], list[float]]:
        if not boards:
            return [], []
        inputs = torch.from_numpy(np.stack([encode_position(board) for board in boards])).to(self.device)
        was_training = self.network.training
        self.network.eval()
        try:
            with torch.inference_mode():
                policy_logits, values = self.network(inputs)
        finally:
            self.network.train(was_training)
        logits = policy_logits.detach().float().cpu().numpy()
        value_list = values.detach().float().cpu().numpy().reshape(-1)
        return [row for row in logits], [max(-1.0, min(1.0, float(v))) for v in value_list]

    @staticmethod
    def _expand(node: _Node, board: chess.Board, logits: np.ndarray) -> None:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            node.expanded = True
            return
        legal_logits = np.asarray([logits[move_to_index(move)] for move in legal_moves], dtype=np.float64)
        legal_logits -= np.max(legal_logits)
        probabilities = np.exp(legal_logits)
        total = probabilities.sum()
        probabilities = probabilities / total if np.isfinite(total) and total > 0 else np.full(len(legal_moves), 1.0 / len(legal_moves))
        node.children = {
            move: _Node(prior=float(probability))
            for move, probability in zip(legal_moves, probabilities, strict=True)
        }
        node.expanded = True

    def _select_child(self, node: _Node) -> tuple[chess.Move, _Node]:
        scale = math.sqrt(max(1, node.visits + node.virtual_visits))
        candidates = [(move, child) for move, child in node.children.items() if not child.pending]
        if not candidates:
            candidates = list(node.children.items())
        return max(candidates, key=lambda item: (
            -((item[1].value_sum - item[1].virtual_visits) / max(1, item[1].visits + item[1].virtual_visits))
            + self.c_puct * item[1].prior * scale / (1 + item[1].visits + item[1].virtual_visits)
        ))

    @staticmethod
    def _backpropagate(path: list[_Node], value: float) -> None:
        for node in reversed(path):
            node.virtual_visits = max(0, node.virtual_visits - 1)
            node.pending = False
            node.visits += 1
            node.value_sum += value
            value = -value

    def search(self, board: chess.Board, *, simulations: int = 100, batch_size: int = 1) -> SearchResult:
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        root_board = board.copy()
        if root_board.is_game_over(claim_draw=True):
            return SearchResult(None, {}, self._terminal_value(root_board))

        root = _Node()
        root_logits, _ = self._evaluate_batch([root_board])
        self._expand(root, root_board, root_logits[0])
        if not root.children:
            return SearchResult(None, {}, self._terminal_value(root_board))

        completed = 0
        while completed < simulations:
            pending: list[tuple[list[_Node], chess.Board]] = []
            terminal: list[tuple[list[_Node], float]] = []
            target = min(batch_size, simulations - completed)
            for _ in range(target):
                current_board = root_board.copy()
                node = root
                path = [root]
                while node.expanded and node.children:
                    move, node = self._select_child(node)
                    current_board.push(move)
                    path.append(node)
                if current_board.is_game_over(claim_draw=True):
                    terminal.append((path, self._terminal_value(current_board)))
                else:
                    pending.append((path, current_board))
                path[-1].pending = True
                for item in path:
                    item.virtual_visits += 1

            if pending:
                logits_batch, values_batch = self._evaluate_batch([item[1] for item in pending])
                for (path, leaf_board), logits, value in zip(pending, logits_batch, values_batch, strict=True):
                    self._expand(path[-1], leaf_board, logits)
                    self._backpropagate(path, value)
            for path, value in terminal:
                self._backpropagate(path, value)
            completed += target

        visit_counts = {move: child.visits for move, child in root.children.items()}
        best_move = max(visit_counts, key=lambda move: visit_counts[move])
        return SearchResult(best_move, visit_counts, root.value)
