# Flora

**Flora** is an experimental chess engine inspired by AlphaZero.

The long-term goal is to combine:
- a neural network that predicts promising moves (policy) and evaluates positions (value),
- Monte Carlo Tree Search (MCTS),
- self-play and reinforcement learning.

This repository starts with the foundations: board encoding, a stable move-index format, a small policy/value network, and tests.

> Flora is an educational experiment, not an AlphaZero-strength engine. Playing strength will be measured through repeatable tests as the project develops.

## Planned milestones

1. Project foundation and environment checks.
2. Chess position and move encoding.
3. Policy/value neural network.
4. Search with MCTS.
5. Self-play data generation.
6. Training, checkpoints, and engine evaluation.

## Requirements

- Python 3.11 or newer
- PyTorch (install the build appropriate for your operating system and CUDA setup)
- The Python dependencies in `pyproject.toml`

## Development setup

Create and activate a virtual environment, then install Flora in editable mode:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Install PyTorch separately using the command recommended by the official PyTorch installation selector:
https://pytorch.org/get-started/locally/

## Current status

Early development. The initial implementation focuses on correctness and clear interfaces before training or search is added.
