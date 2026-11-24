# Flora

**Flora** is an experimental neural chess engine inspired by AlphaZero. It combines a policy/value network, batched neural MCTS, self-play data generation, and supervised policy/value training. A classical alpha-beta engine remains available as a baseline.

> Flora is an educational experiment. Its playing strength has not yet been established.

## Current features

- 18-plane board encoding and fixed move vocabulary
- PyTorch residual policy/value network
- PUCT Monte Carlo Tree Search with batched neural inference
- Self-play game generation with MCTS visit-count policy targets
- PGN game importer for supervised policy/value pretraining
- Compressed NPZ training datasets
- Policy KL-divergence and value mean-squared-error training
- Classical alpha-beta search baseline and UCI protocol

The network starts untrained. Its initial predictions are effectively random and must not be interpreted as chess knowledge. Training on a tiny self-play sample is only a pipeline smoke test, not meaningful strength training.

## Install

Use Python 3.11+ and install the project with its training dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,training]"
```

Install a PyTorch build appropriate for your CUDA setup from https://pytorch.org/get-started/locally/.

## Generate self-play data

```powershell
python -m flora.selfplay --games 2 --simulations 64 --batch-size 8 --device cuda --output data/selfplay.npz
```

To use a previously trained model:

```powershell
python -m flora.selfplay --checkpoint checkpoints/flora.pt --games 10 --simulations 128 --batch-size 16 --device cuda
```

Self-play games stop at the normal game-ending rules or the configured maximum ply count. Games truncated by the ply limit are labelled as draws, so increase the limit for serious data generation.

## Supervised pretraining from PGN

Place one or more `.pgn` files in a folder, or provide a single PGN file:

```powershell
python -m flora.pgn --input data/pgn --output data/pgn_training.npz
```

For a small test, limit the import:

```powershell
python -m flora.pgn --input data/games.pgn --max-games 100 --max-plies 100 --output data/pgn_training.npz
```

Only games with a known result (`1-0`, `0-1`, or `1/2-1/2`) are included. The played move is the policy target; the final game result is the value target from the side-to-move perspective. This is basic outcome-supervised learning, not an engine evaluation label.

## Train a network

```powershell
python -m flora.train --data data/pgn_training.npz --epochs 5 --batch-size 32 --device cuda --output checkpoints/flora.pt
```

To initialize from an existing Flora model and fine-tune its weights:

```powershell
python -m flora.train --data data/pgn_training.npz --checkpoint checkpoints/previous.pt --epochs 5 --device cuda --output checkpoints/flora_finetuned.pt
```

This resumes model weights, but currently starts a fresh optimizer; optimizer state and training epoch are not resumed.

## Run the classical UCI baseline

```powershell
python -m flora.uci
```

The existing UCI engine uses classical alpha-beta search. Neural MCTS is currently exposed as a Python API and self-play tool; it is not yet connected to UCI. True background pondering is also not implemented.

## Tests

```powershell
pytest
```

## Roadmap

1. Validate batched MCTS and training-data correctness.
2. Add checkpoint resume, replay-buffer management, and model-vs-model evaluation.
3. Improve self-play sampling, draw handling, and training stability.
4. Add a neural-MCTS UCI mode with time/simulation controls.
5. Measure playing strength against fixed baselines before making strength claims.
