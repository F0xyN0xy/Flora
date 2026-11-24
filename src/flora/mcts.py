"""Monte Carlo Tree Search guided by Flora's policy/value network."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess
import numpy as np
import torch

from flora.encoding import encode_position, move_to_index


@dataclass
class _Node:
    """A node's value statistics are always from its side-to-move perspective."""

    prior: float = 1.0
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False
    children: dict[chess.Move, "_Node"] = field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class SearchResult:
    """Summary of a completed search at the root position."""

    best_move: chess.Move | None
    visit_counts: dict[chess.Move, int]
    root_value: float


class MCTS:
    """PUCT search using a policy/value network.

    The network's value is interpreted from the perspective of the player
    whose turn it is in the supplied board position.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        *,
        device: str | torch.device = "cpu",
        c_puct: float = 1.5,
    ) -> None:
        if c_puct <= 0:
            raise ValueError("c_puct must be greater than zero")

        self.network = network
        self.device = torch.device(device)
        self.c_puct = c_puct
        self.network.to(self.device)

    def _evaluate(self, board: chess.Board) -> tuple[np.ndarray, float]:
        encoded = encode_position(board)
        inputs = torch.from_numpy(encoded).unsqueeze(0).to(self.device)

        was_training = self.network.training
        self.network.eval()
        try:
            with torch.inference_mode():
                policy_logits, values = self.network(inputs)
        finally:
            self.network.train(was_training)

        logits = policy_logits[0].detach().float().cpu().numpy()
        value = float(values.reshape(-1)[0].detach().cpu().item())
        return logits, max(-1.0, min(1.0, value))

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        """Return terminal value from the current side-to-move perspective."""
        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            raise ValueError("Position is not terminal")
        if outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0

    def _expand(self, node: _Node, board: chess.Board) -> float:
        if board.is_game_over(claim_draw=True):
            node.expanded = True
            return self._terminal_value(board)

        logits, value = self._evaluate(board)
        legal_moves = list(board.legal_moves)
        legal_logits = np.asarray(
            [logits[move_to_index(move)] for move in legal_moves],
            dtype=np.float64,
        )

        # Stable softmax over legal moves only; illegal moves get no prior.
        legal_logits -= np.max(legal_logits)
        probabilities = np.exp(legal_logits)
        probabilities /= probabilities.sum()

        node.children = {
            move: _Node(prior=float(probability))
            for move, probability in zip(legal_moves, probabilities, strict=True)
        }
        node.expanded = True
        return value

    def _select_child(self, node: _Node) -> tuple[chess.Move, _Node]:
        exploration_scale = math.sqrt(max(1, node.visits))
        return max(
            node.children.items(),
            key=lambda item: (
                -item[1].value
                + self.c_puct
                * item[1].prior
                * exploration_scale
                / (1 + item[1].visits)
            ),
        )

    @staticmethod
    def _backpropagate(path: list[_Node], value: float) -> None:
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = -value

    def search(self, board: chess.Board, *, simulations: int = 100) -> SearchResult:
        """Run PUCT simulations and return the most-visited root move."""
        if simulations < 1:
            raise ValueError("simulations must be at least 1")

        root_board = board.copy()
        root = _Node()
        self._expand(root, root_board)

        if not root.children:
            return SearchResult(best_move=None, visit_counts={}, root_value=0.0)

        for _ in range(simulations):
            current_board = root_board.copy()
            node = root
            path = [root]

            while node.expanded and node.children:
                move, node = self._select_child(node)
                current_board.push(move)
                path.append(node)

            leaf_value = self._expand(node, current_board)
            self._backpropagate(path, leaf_value)

        visit_counts = {
            move: child.visits for move, child in root.children.items()
        }
        best_move = max(visit_counts, key=visit_counts.get)
        return SearchResult(
            best_move=best_move,
            visit_counts=visit_counts,
            root_value=root.value,
        )
