"""Iterative-deepening alpha-beta search for Flora."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable

import chess

from flora.evaluation import evaluate

MATE_SCORE = 100_000
INF = 200_000


@dataclass(frozen=True)
class SearchInfo:
    depth: int
    score: int
    nodes: int
    elapsed: float
    pv: tuple[chess.Move, ...]


@dataclass(frozen=True)
class SearchResult:
    best_move: chess.Move | None
    ponder_move: chess.Move | None
    score: int
    depth: int
    nodes: int
    elapsed: float
    pv: tuple[chess.Move, ...]


class _SearchStopped(Exception):
    pass


class Searcher:
    """Alpha-beta search with iterative deepening, quiescence and a TT."""

    def __init__(self) -> None:
        self.nodes = 0
        self.table: dict[tuple, tuple[int, int, int, chess.Move | None]] = {}
        self.stop_event = threading.Event()
        self.deadline: float | None = None
        self.node_limit: int | None = None
        self.info_callback: Callable[[SearchInfo], None] | None = None
        self.killers: dict[int, list[chess.Move]] = {}
        self.history: dict[tuple[bool, chess.Move], int] = {}

    @staticmethod
    def _key(board: chess.Board) -> tuple:
        return (
            board.board_fen(),
            board.turn,
            board.castling_rights,
            board.ep_square,
            board.halfmove_clock,
        )

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise _SearchStopped
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.stop_event.set()
            raise _SearchStopped
        if self.node_limit is not None and self.nodes >= self.node_limit:
            self.stop_event.set()
            raise _SearchStopped

    def _move_order_score(
        self,
        board: chess.Board,
        move: chess.Move,
        tt_move: chess.Move | None,
        ply: int,
    ) -> int:
        if move == tt_move:
            return 2_000_000
        if move.promotion:
            return 1_000_000 + move.promotion * 100
        if board.is_capture(move):
            victim = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
            attacker = board.piece_type_at(move.from_square)
            return 100_000 + 10 * PIECE_ORDER_VALUE.get(victim, 0) - PIECE_ORDER_VALUE.get(attacker, 0)
        killers = self.killers.get(ply, [])
        if killers and move == killers[0]:
            return 90_000
        if len(killers) > 1 and move == killers[1]:
            return 80_000
        return self.history.get((board.turn, move), 0)

    def _ordered_moves(
        self,
        board: chess.Board,
        tt_move: chess.Move | None = None,
        ply: int = 0,
        moves: list[chess.Move] | None = None,
    ) -> list[chess.Move]:
        candidates = list(board.legal_moves) if moves is None else moves
        candidates.sort(
            key=lambda move: self._move_order_score(board, move, tt_move, ply),
            reverse=True,
        )
        return candidates

    def _record_quiet_cutoff(self, board: chess.Board, move: chess.Move, ply: int, depth: int) -> None:
        if board.is_capture(move) or move.promotion:
            return
        killers = self.killers.setdefault(ply, [])
        if not killers or killers[0] != move:
            killers.insert(0, move)
            del killers[2:]
        key = (board.turn, move)
        self.history[key] = min(1_000_000, self.history.get(key, 0) + depth * depth)

    def _quiescence(self, board: chess.Board, alpha: int, beta: int, ply: int) -> int:
        self.nodes += 1
        self._check_stop()
        if board.is_checkmate():
            return -MATE_SCORE + ply
        if board.is_game_over(claim_draw=True):
            return 0

        in_check = board.is_check()
        if not in_check:
            stand_pat = evaluate(board)
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
            legal_moves = list(board.legal_moves)
            moves = [
                move for move in legal_moves
                if board.is_capture(move) or move.promotion
            ]
        else:
            moves = list(board.legal_moves)

        for move in self._ordered_moves(board, ply=ply, moves=moves):
            board.push(move)
            try:
                score = -self._quiescence(board, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
        return alpha

    def _negamax(
        self, board: chess.Board, depth: int, alpha: int, beta: int, ply: int
    ) -> tuple[int, list[chess.Move]]:
        self.nodes += 1
        self._check_stop()
        if board.is_checkmate():
            return -MATE_SCORE + ply, []
        if board.is_game_over(claim_draw=True):
            return 0, []
        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply), []

        original_alpha = alpha
        original_beta = beta
        key = self._key(board)
        entry = self.table.get(key)
        tt_move = entry[3] if entry else None
        if entry and entry[0] >= depth:
            tt_score, flag = entry[1], entry[2]
            if flag == 0:
                return tt_score, [tt_move] if tt_move else []
            if flag == 1:
                alpha = max(alpha, tt_score)
            elif flag == 2:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score, [tt_move] if tt_move else []

        best_score = -INF
        best_move = None
        best_pv: list[chess.Move] = []
        for move in self._ordered_moves(board, tt_move, ply):
            board.push(move)
            try:
                score, child_pv = self._negamax(board, depth - 1, -beta, -alpha, ply + 1)
            finally:
                board.pop()
            score = -score
            if score > best_score:
                best_score, best_move = score, move
                best_pv = [move, *child_pv]
            alpha = max(alpha, score)
            if alpha >= beta:
                self._record_quiet_cutoff(board, move, ply, depth)
                break

        flag = 2 if best_score <= original_alpha else 1 if best_score >= original_beta else 0
        self.table[key] = (depth, best_score, flag, best_move)
        return best_score, best_pv

    def search(
        self,
        board: chess.Board,
        *,
        max_depth: int = 6,
        time_limit: float | None = None,
        node_limit: int | None = None,
        info_callback: Callable[[SearchInfo], None] | None = None,
    ) -> SearchResult:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if time_limit is not None and time_limit <= 0:
            raise ValueError("time_limit must be positive")
        if node_limit is not None and node_limit < 1:
            raise ValueError("node_limit must be at least 1")

        root = board.copy()
        self.nodes = 0
        self.table.clear()
        self.killers.clear()
        self.history.clear()
        self.stop_event.clear()
        self.deadline = time.monotonic() + time_limit if time_limit is not None else None
        self.node_limit = node_limit
        self.info_callback = info_callback
        start = time.monotonic()

        legal = list(root.legal_moves)
        if not legal:
            return SearchResult(None, None, -MATE_SCORE if root.is_checkmate() else 0,
                                0, 0, 0.0, ())

        best_move = legal[0]
        best_score = evaluate(root)
        best_pv: list[chess.Move] = [best_move]
        completed_depth = 0
        for depth in range(1, max_depth + 1):
            try:
                score, pv = self._negamax(root, depth, -INF, INF, 0)
            except _SearchStopped:
                break
            if pv:
                best_move, best_pv = pv[0], pv
                best_score = score
            completed_depth = depth
            elapsed = time.monotonic() - start
            if info_callback:
                info_callback(SearchInfo(depth, best_score, self.nodes, elapsed, tuple(best_pv)))
            if abs(best_score) >= MATE_SCORE - depth:
                break

        elapsed = time.monotonic() - start
        ponder = best_pv[1] if len(best_pv) > 1 else None
        return SearchResult(best_move, ponder, best_score, completed_depth,
                            self.nodes, elapsed, tuple(best_pv))


PIECE_ORDER_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 10,
}
