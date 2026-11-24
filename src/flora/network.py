"""A compact convolutional policy/value network for Flora."""

from __future__ import annotations

import torch
from torch import nn

from flora.encoding import MOVE_VOCABULARY_SIZE, POSITION_PLANES


class ResidualBlock(nn.Module):
    """Two-convolution residual block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class FloraNetwork(nn.Module):
    """Predict move logits and a scalar position value.

    The policy output uses Flora's fixed move vocabulary. Illegal moves are
    filtered later by the search code, not by this network.
    """

    def __init__(
        self,
        channels: int = 64,
        residual_blocks: int = 4,
    ) -> None:
        super().__init__()
        if channels < 8:
            raise ValueError("channels must be at least 8")
        if residual_blocks < 1:
            raise ValueError("residual_blocks must be at least 1")

        self.stem = nn.Sequential(
            nn.Conv2d(POSITION_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.backbone = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(residual_blocks))
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, MOVE_VOCABULARY_SIZE),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(4 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits, values) for a batch of encoded positions."""
        if positions.ndim != 4 or positions.shape[1:] != (POSITION_PLANES, 8, 8):
            raise ValueError(
                f"Expected input shape (batch, {POSITION_PLANES}, 8, 8), "
                f"got {tuple(positions.shape)}"
            )

        features = self.backbone(self.stem(positions))
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_logits, values
