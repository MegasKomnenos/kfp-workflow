"""SOFTS RUL model: SOFTSForRUL wrapper, build_model, and device configuration."""

from __future__ import annotations

import types
from typing import Dict

import torch
import torch.nn as nn


def configure_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA device if available and requested, otherwise CPU."""
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")


def _make_softs_configs(cfg: Dict, seq_len: int) -> types.SimpleNamespace:
    """Build a SimpleNamespace config object for SOFTS.Model from a flat cfg dict.

    pred_len is always 1 (RUL is a scalar per window).
    use_norm defaults to False because the C-MAPSS preprocessing pipeline already
    applies condition-based normalization — a second in-model normalization would
    corrupt the signal.
    """
    return types.SimpleNamespace(
        seq_len=seq_len,
        pred_len=1,
        d_model=int(cfg.get("d_model", 64)),
        d_core=int(cfg.get("d_core", 32)),
        d_ff=int(cfg.get("d_ff", 256)),
        e_layers=int(cfg.get("e_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        activation=str(cfg.get("activation", "gelu")),
        use_norm=bool(cfg.get("use_norm", False)),
    )


class SOFTSForRUL(nn.Module):
    """SOFTS backbone adapted for scalar RUL regression.

    The SOFTS backbone produces ``[B, pred_len=1, c_in]``.  A linear head
    projects from ``c_in`` to ``1``, yielding a scalar prediction per sample.

    Parameters
    ----------
    configs:
        SimpleNamespace with SOFTS hyperparameters (see ``_make_softs_configs``).
    c_in:
        Number of input features (sensor + operational columns).
    """

    def __init__(self, configs: types.SimpleNamespace, c_in: int):
        super().__init__()
        from softs_new.softs_layers.softs import Model as SOFTSModel
        self.backbone = SOFTSModel(configs)
        self.head = nn.Linear(c_in, 1)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return RUL predictions.

        Parameters
        ----------
        x:
            Input tensor ``[B, seq_len, c_in]``.

        Returns
        -------
        torch.Tensor
            Predictions ``[B, 1]``.
        """
        out = self.backbone(x, None, None, None)   # [B, 1, c_in]
        return self.head(out.squeeze(1))  # [B, 1]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    cfg: Dict,
    c_in: int,
    seq_len: int,
    device: torch.device,
) -> SOFTSForRUL:
    """Instantiate ``SOFTSForRUL`` from a flat config dict and move to device.

    Parameters
    ----------
    cfg:
        Flat config dict containing SOFTS hyperparameters.
    c_in:
        Number of input features.
    seq_len:
        Window / sequence length.
    device:
        Target torch device.
    """
    configs = _make_softs_configs(cfg, seq_len)
    return SOFTSForRUL(configs, c_in).to(device)
