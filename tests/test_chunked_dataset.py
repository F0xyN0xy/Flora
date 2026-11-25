import json

import numpy as np

from flora.dataset import FloraDataset, list_dataset_chunks, write_chunked_dataset


def _example(value):
    position = np.zeros((18, 8, 8), dtype=np.float32)
    policy = np.zeros(20480, dtype=np.float32)
    policy[value] = 1.0
    return position, policy, float(value % 2)


def test_chunked_writer_splits_examples_and_writes_manifest(tmp_path):
    output = tmp_path / "training"
    total, chunks = write_chunked_dataset(
        output, (_example(i) for i in range(5)), chunk_size=2
    )

    assert (total, chunks) == (5, 3)
    assert len(list_dataset_chunks(output)) == 3
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["positions"] == 5
    assert manifest["chunks"] == 3
    assert [len(FloraDataset(path)) for path in list_dataset_chunks(output)] == [2, 2, 1]


def test_chunked_writer_refuses_to_overwrite_existing_chunks(tmp_path):
    output = tmp_path / "training"
    write_chunked_dataset(output, [_example(0)], chunk_size=1)

    try:
        write_chunked_dataset(output, [_example(1)], chunk_size=1)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Expected existing chunks to be protected")


def test_list_dataset_chunks_accepts_legacy_npz(tmp_path):
    from flora.dataset import save_examples

    path = tmp_path / "legacy.npz"
    save_examples(path, [_example(0)])
    assert list_dataset_chunks(path) == [path]


def test_chunked_writer_preserves_game_ids_across_shards(tmp_path):
    output = tmp_path / "game-aware"
    examples = [
        (_example(i)[0], _example(i)[1], 0.0, game_id)
        for i, game_id in enumerate([10, 10, 11, 11, 12])
    ]
    total, chunks = write_chunked_dataset(output, examples, chunk_size=2)

    assert (total, chunks) == (5, 3)
    observed = []
    for path in list_dataset_chunks(output):
        with np.load(path, allow_pickle=False) as data:
            observed.extend(data["game_ids"].tolist())
    assert observed == [10, 10, 11, 11, 12]

    dataset = FloraDataset(list_dataset_chunks(output)[0], indices=[1])
    assert len(dataset) == 1
