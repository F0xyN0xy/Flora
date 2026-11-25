# Flora

**Flora** is an experimental neural chess engine inspired by AlphaZero. It combines a policy/value network, batched neural MCTS, self-play data generation, and supervised policy/value training. A classical alpha-beta engine remains available as a baseline.

> Flora is an educational experiment. Its playing strength has not yet been established.

## Current features

- 18-plane board encoding and fixed move vocabulary
- PyTorch residual policy/value network
- PUCT Monte Carlo Tree Search with batched neural inference
- Self-play game generation with MCTS visit-count policy targets
- Automatic PGN recording of self-play games
- PGN importer that streams positions into bounded-size dataset shards
- Chunk-by-chunk training to limit host RAM usage
- Policy KL-divergence and value mean-squared-error training
- Classical alpha-beta search baseline and UCI protocol
- Experimental neural-MCTS arena against an external UCI engine

The network starts untrained. Its initial predictions are effectively random and must not be interpreted as chess knowledge. Training on a tiny self-play sample is only a pipeline smoke test, not meaningful strength training.

## Install

Use Python 3.11+ and install the project with its training dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,training]"
```

Install a PyTorch build appropriate for your CUDA setup from https://pytorch.org/get-started/locally/.

## Generate self-play games

```powershell
python -m flora.selfplay --games 2 --simulations 64 --batch-size 8 --device cuda --output data/selfplay.npz --pgn-output data/selfplay_games.pgn
```

Every game is appended to the PGN file as it finishes. To use a previously trained model:

```powershell
python -m flora.selfplay --checkpoint checkpoints/flora.pt --games 10 --simulations 128 --batch-size 16 --device cuda
```

Self-play games stop at the normal game-ending rules or the configured maximum ply count. A game cut off by the ply limit is saved as unfinished (`*`) in the PGN and its positions currently receive draw value targets. Increase the limit for serious data generation.

## Supervised pretraining from large PGN collections

Place one or more `.pgn` files in a folder, or provide a single PGN file:

```powershell
python -m flora.pgn --input data/pgn --output data/pgn_training --chunk-size 256
```

The importer reads one game at a time, records a stable game ID with every position, and writes a shard after every 256 positions by default. Game IDs are preserved even when one game spans multiple shards. It does not retain the entire PGN collection or all training examples in RAM. Choose a new or empty output directory; the importer deliberately refuses to overwrite existing chunk files.

For a small test:

```powershell
python -m flora.pgn --input data/pgn --max-games 100 --max-plies 100 --output data/pgn_test --chunk-size 128
```

Only games with a known result (`1-0`, `0-1`, or `1/2-1/2`) are included. The played move is the policy target; the final game result is the value target from the side-to-move perspective. This is outcome-supervised learning, not an engine evaluation label.

## Train a network

The trainer accepts either a legacy single NPZ file or a chunked dataset directory. New PGN-imported datasets use game-aware validation: all positions from a given game are assigned to either training or validation, never both. Older datasets without game IDs remain readable, but fall back to shard-based validation with a warning; re-import the PGNs to get game-aware validation. With chunked data, it loads one shard at a time:

```powershell
python -m flora.train --data data/pgn_training --epochs 5 --batch-size 32 --device cuda --output checkpoints/flora.pt
```

To initialize from an existing Flora model and fine-tune its weights:

```powershell
python -m flora.train --data data/pgn_training --checkpoint checkpoints/previous.pt --epochs 5 --device cuda --output checkpoints/flora_finetuned.pt
```

Training checkpoints preserve model weights, optimizer state, completed epoch, loss history, and best validation loss. Use `--checkpoint` to resume for the requested number of additional epochs. RAM use is bounded by the largest individual shard, not the entire dataset.

## Play against another UCI engine

Flora's experimental neural-MCTS arena can launch an external UCI engine executable and automatically append each game to a PGN:

```powershell
python -m flora.arena --checkpoint checkpoints/flora.pt --opponent "C:\\Engines\\opponent.exe" --games 10 --simulations 128 --batch-size 8 --opponent-time 0.25 --flora-color alternate --device cuda
```

The opponent must be a local executable that speaks the UCI protocol. This arena uses Flora's neural MCTS directly; it is separate from the current classical UCI engine. The neural checkpoint must exist and match Flora's current network architecture. The opponent time is seconds per move. Results are experimental and depend heavily on checkpoint quality and search settings.

## Run the classical UCI baseline

```powershell
python -m flora.uci
```

The existing UCI engine uses classical alpha-beta search. Neural MCTS is not yet connected to the UCI protocol, and true background pondering is not implemented.

## Tests

```powershell
pytest
```

## Roadmap

1. Validate chunked PGN import and training on large collections.
2. Improve self-play sampling, draw handling, and training stability.
3. Evaluate neural MCTS against fixed external engines and track results.
4. Add a neural-MCTS UCI mode with time/simulation controls.
5. Package a tested neural UCI engine for chess GUIs.
