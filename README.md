# Flora

**Flora** is an experimental chess engine inspired by AlphaZero. The project combines a transparent classical search baseline with a policy/value neural network, MCTS, and a planned self-play training pipeline.

> Flora is an educational experiment. Its playing strength is not yet established.

## Current features

- Chess position and move encoding
- PyTorch policy/value network
- Policy-guided MCTS
- Classical material/positional evaluation
- Iterative-deepening negamax with alpha-beta pruning
- Quiescence search and a transposition table
- UCI protocol support with search info and principal variation output

The classical search is currently the usable playing baseline. The neural network is not trained yet; its output should not be interpreted as meaningful chess knowledge.

## Requirements

- Python 3.11 or newer
- PyTorch for the neural network and MCTS
- The Python dependencies in `pyproject.toml`

## Development setup

Create and activate a virtual environment, then install Flora in editable mode:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,training]"
```

Install the PyTorch build appropriate for your operating system and CUDA setup using the official selector:
https://pytorch.org/get-started/locally/

## Run Flora

Start the UCI engine from the project directory:

```powershell
python -m flora.uci
```

Or, after installing the project in editable mode:

```powershell
flora-uci
```

Flora currently supports the UCI handshake, readiness checks, new games, positions using `startpos` or FEN, and `go depth`, `go movetime`, `go nodes`, and basic clock-based `go wtime/btime` searches. During search it reports depth, score, nodes, elapsed time, nodes per second, and its principal variation. Its final `bestmove` may include the predicted reply as a UCI `ponder` move.

Actual background pondering (continuing to search the predicted reply while waiting for the opponent) is not implemented yet.

## Tests

Run:

```powershell
pytest
```

## Roadmap

1. Validate classical evaluation and UCI behaviour against chess positions and GUIs.
2. Improve search strength and add engine-vs-engine benchmarks.
3. Add self-play game generation and training-data storage.
4. Train policy/value checkpoints and evaluate them against previous versions.
5. Integrate trained neural guidance into search and explore true UCI pondering.
