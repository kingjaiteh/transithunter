"""AstroNet-style two-branch 1D CNN (Shallue & Vanderburg 2018).

The global branch reads the whole folded orbit, the local branch reads the
zoom on the transit. Their flattened outputs are concatenated and passed
through fully connected layers to a single logit. Widths are configurable;
the default is narrower than the paper's because dataset v1 has about 3,000
examples rather than 15,000.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
from torch import nn

from transithunter import config


@dataclass
class CNNConfig:
    global_channels: tuple[int, ...] = (16, 32, 64, 128)
    local_channels: tuple[int, ...] = (16, 32)
    kernel: int = 5
    global_pool: tuple[int, int] = (5, 2)  # (kernel, stride)
    local_pool: tuple[int, int] = (7, 2)
    hidden: tuple[int, ...] = (256, 256)
    dropout: float = 0.3
    n_aux: int = 0  # allowlisted scalar features appended before the FC head, 0 = views only
    global_bins: int = config.GLOBAL_BINS
    local_bins: int = config.LOCAL_BINS

    def to_dict(self) -> dict:
        return asdict(self)


def _branch(channels: tuple[int, ...], kernel: int, pool: tuple[int, int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    c_in = 1
    for c_out in channels:
        layers += [
            nn.Conv1d(c_in, c_out, kernel, padding=kernel // 2), nn.ReLU(),
            nn.Conv1d(c_out, c_out, kernel, padding=kernel // 2), nn.ReLU(),
            nn.MaxPool1d(pool[0], stride=pool[1]),
        ]
        c_in = c_out
    return nn.Sequential(*layers)


class TransitCNN(nn.Module):
    def __init__(self, cfg: CNNConfig | None = None):
        super().__init__()
        self.cfg = cfg or CNNConfig()
        c = self.cfg
        self.global_branch = _branch(c.global_channels, c.kernel, c.global_pool)
        self.local_branch = _branch(c.local_channels, c.kernel, c.local_pool)
        with torch.no_grad():
            n_flat = (self.global_branch(torch.zeros(1, 1, c.global_bins)).numel()
                      + self.local_branch(torch.zeros(1, 1, c.local_bins)).numel())
        head: list[nn.Module] = []
        n_in = n_flat + c.n_aux
        for h in c.hidden:
            head += [nn.Linear(n_in, h), nn.ReLU(), nn.Dropout(c.dropout)]
            n_in = h
        head.append(nn.Linear(n_in, 1))
        self.head = nn.Sequential(*head)

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor,
                aux: torch.Tensor | None = None) -> torch.Tensor:
        """Views are (B, bins). Returns logits (B,)."""
        g = self.global_branch(global_view.unsqueeze(1)).flatten(1)
        l = self.local_branch(local_view.unsqueeze(1)).flatten(1)
        parts = [g, l] if aux is None else [g, l, aux]
        return self.head(torch.cat(parts, dim=1)).squeeze(1)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
