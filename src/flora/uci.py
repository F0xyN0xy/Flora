"""UCI protocol entry point for connecting Flora to chess GUIs."""

from __future__ import annotations

import sys
import threading
import time

import chess

from flora.engine import SearchInfo, Searcher


class UCIEngine:
    def __init__(self) -> None:
        self.board = chess.Board()
        self.searcher = Searcher()
        self.worker: threading.Thread | None = None
        self.output_lock = threading.Lock()
        self.quit_event = threading.Event()

    def send(self, line: str) -> None:
        with self.output_lock:
            print(line, flush=True)

    @staticmethod
    def _uci_score(score: int) -> str:
        if abs(score) >= 99_000:
            plies = max(1, 100_000 - abs(score))
            moves = max(1, (plies + 1) // 2)
            return f"mate {moves if score > 0 else -moves}"
        return f"cp {score}"

    def _info(self, info: SearchInfo) -> None:
        elapsed_ms = max(1, int(info.elapsed * 1000))
        nps = int(info.nodes / max(info.elapsed, 0.001))
        pv = " ".join(move.uci() for move in info.pv)
        self.send(
            f"info depth {info.depth} score {self._uci_score(info.score)} "
            f"nodes {info.nodes} time {elapsed_ms} nps {nps} pv {pv}"
        )

    def _run_search(self, limits: dict[str, float | int]) -> None:
        kwargs: dict = {"max_depth": int(limits.get("depth", 64)), "info_callback": self._info}
        if "movetime" in limits:
            kwargs["time_limit"] = max(0.01, float(limits["movetime"]) / 1000)
        if "nodes" in limits:
            kwargs["node_limit"] = int(limits["nodes"])
        result = self.searcher.search(self.board, **kwargs)
        if result.best_move is None:
            self.send("bestmove 0000")
            return
        line = f"bestmove {result.best_move.uci()}"
        if result.ponder_move is not None:
            line += f" ponder {result.ponder_move.uci()}"
        self.send(line)

    def _stop_worker(self) -> None:
        self.searcher.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join()
        self.worker = None

    def _start_search(self, limits: dict[str, float | int]) -> None:
        self._stop_worker()
        self.worker = threading.Thread(target=self._run_search, args=(limits,), daemon=True)
        self.worker.start()

    def handle(self, command: str) -> bool:
        parts = command.strip().split()
        if not parts:
            return True
        name = parts[0]

        if name == "uci":
            self.send("id name Flora")
            self.send("id author F0xyN0xy")
            self.send("uciok")
        elif name == "isready":
            self.send("readyok")
        elif name == "ucinewgame":
            self._stop_worker()
            self.board = chess.Board()
        elif name == "position":
            self._stop_worker()
            try:
                if parts[1] == "startpos":
                    board = chess.Board()
                    index = 2
                elif parts[1] == "fen":
                    marker = parts.index("moves") if "moves" in parts else len(parts)
                    board = chess.Board(" ".join(parts[2:marker]))
                    index = marker
                else:
                    raise ValueError("expected startpos or fen")
                if index < len(parts) and parts[index] == "moves":
                    for token in parts[index + 1:]:
                        move = chess.Move.from_uci(token)
                        if move not in board.legal_moves:
                            raise ValueError(f"illegal move: {token}")
                        board.push(move)
                self.board = board
            except (ValueError, IndexError) as exc:
                self.send(f"info string invalid position: {exc}")
        elif name == "go":
            limits: dict[str, float | int] = {}
            tokens = parts[1:]
            for i, token in enumerate(tokens[:-1]):
                if token in ("depth", "movetime", "nodes", "wtime", "btime", "winc", "binc"):
                    limits[token] = float(tokens[i + 1]) if token == "movetime" else int(tokens[i + 1])
            if "movetime" not in limits:
                clock_key = "wtime" if self.board.turn == chess.WHITE else "btime"
                inc_key = "winc" if self.board.turn == chess.WHITE else "binc"
                if clock_key in limits:
                    remaining = max(1, int(limits[clock_key]))
                    increment = int(limits.get(inc_key, 0))
                    limits["movetime"] = max(10, min(remaining // 2, remaining // 30 + increment * 4 // 5))
            self._start_search(limits)
        elif name == "stop":
            self._stop_worker()
        elif name == "quit":
            self._stop_worker()
            self.quit_event.set()
            return False
        elif name == "setoption":
            self.send("info string no configurable options yet")
        return True

    def run(self) -> None:
        for line in sys.stdin:
            if not self.handle(line):
                break
        self._stop_worker()


def main() -> None:
    UCIEngine().run()


if __name__ == "__main__":
    main()
