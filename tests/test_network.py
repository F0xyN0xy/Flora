import pytest

torch = pytest.importorskip("torch")

from flora.encoding import MOVE_VOCABULARY_SIZE, POSITION_PLANES
from flora.network import FloraNetwork


def test_network_output_shapes():
    model = FloraNetwork(channels=16, residual_blocks=1)
    positions = torch.zeros((2, POSITION_PLANES, 8, 8))
    policy, value = model(positions)

    assert policy.shape == (2, MOVE_VOCABULARY_SIZE)
    assert value.shape == (2,)
    assert torch.all(value >= -1.0)
    assert torch.all(value <= 1.0)


def test_network_rejects_invalid_input_shape():
    model = FloraNetwork(channels=16, residual_blocks=1)
    with pytest.raises(ValueError):
        model(torch.zeros((1, 12, 8, 8)))
