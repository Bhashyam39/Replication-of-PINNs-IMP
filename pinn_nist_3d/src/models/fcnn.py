"""Fully-connected neural network backbone for the 3D NIST PINN.

PyTorch implementation (paper used TensorFlow; see SPEC "Hard deviations").

Interface contract (SPEC):
    forward accepts a tensor of shape (N, 4) ordered (x, y, z, t) [SI: m, m, m, s]
    and returns (N, 5) ordered (u, v, w, p, T) [m/s, m/s, m/s, Pa, K].
"""
from __future__ import annotations

import torch
from torch import nn

__all__ = ["Swish", "FCNN", "init_weights"]

#: Fixed output column order required by the SPEC interface contract.
OUTPUT_ORDER = ("u", "v", "w", "p", "T")
#: Fixed input column order required by the SPEC interface contract.
INPUT_ORDER = ("x", "y", "z", "t")


class Swish(nn.Module):
    """Swish activation: f(x) = x * sigmoid(x) (a.k.a. SiLU)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x * torch.sigmoid(x)


def init_weights(module: nn.Module) -> None:
    """Xavier (Glorot) uniform initialization for Linear layers.

    Biases are initialized to zero. Intended for use with
    ``model.apply(init_weights)``.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FCNN(nn.Module):
    """Fully-connected PINN backbone with Swish activations.

    Parameters
    ----------
    in_dim:
        Input dimension. Default 4 -> (x, y, z, t).
    out_dim:
        Output dimension. Default 5 -> (u, v, w, p, T).
    hidden:
        Width of each hidden layer. Default 250 (paper-style width).
    layers:
        Number of hidden layers. Default 5.
    """

    def __init__(
        self,
        in_dim: int = 4,
        out_dim: int = 5,
        hidden: int = 250,
        layers: int = 5,
    ) -> None:
        super().__init__()
        if in_dim < 1 or out_dim < 1:
            raise ValueError("in_dim and out_dim must be positive")
        if hidden < 1:
            raise ValueError("hidden must be positive")
        if layers < 1:
            raise ValueError("layers must be >= 1")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden = hidden
        self.layers = layers

        blocks: list[nn.Module] = []
        prev = in_dim
        for _ in range(layers):
            blocks.append(nn.Linear(prev, hidden))
            blocks.append(Swish())
            prev = hidden
        blocks.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*blocks)

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map (N, 4) inputs (x, y, z, t) to (N, 5) outputs (u, v, w, p, T)."""
        if x.dim() != 2 or x.size(-1) != self.in_dim:
            raise ValueError(
                f"FCNN expects input of shape (N, {self.in_dim}) ordered "
                f"{INPUT_ORDER}, got {tuple(x.shape)}"
            )
        return self.net(x)
