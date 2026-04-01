"""TimeMixer RUL model: TimeMixerForRUL wrapper, build_model, and device configuration."""

from __future__ import annotations

import types
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def configure_device(prefer_gpu: bool = True) -> torch.device:
    """Return CUDA device if available and requested, otherwise CPU."""
    return torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")


def _make_timemixer_configs(cfg: Dict, c_in: int, seq_len: int) -> types.SimpleNamespace:
    """Build a SimpleNamespace config object for TimeMixer.Model from a flat cfg dict.

    C-MAPSS preprocessing already applies condition-based normalization, so
    ``use_norm`` defaults to 0 (disabled) to avoid double normalization.
    """
    return types.SimpleNamespace(
        seq_len=seq_len,
        enc_in=c_in,
        d_model=int(cfg.get("d_model", 64)),
        d_ff=int(cfg.get("d_ff", 256)),
        e_layers=int(cfg.get("e_layers", 2)),
        dropout=float(cfg.get("dropout", 0.1)),
        channel_independence=int(cfg.get("channel_independence", 1)),
        down_sampling_layers=int(cfg.get("down_sampling_layers", 2)),
        down_sampling_window=int(cfg.get("down_sampling_window", 2)),
        down_sampling_method=str(cfg.get("down_sampling_method", "avg")),
        decomp_method=str(cfg.get("decomp_method", "moving_avg")),
        moving_avg=int(cfg.get("moving_avg", 25)),
        top_k=int(cfg.get("top_k", 5)),
        use_norm=int(cfg.get("use_norm", 0)),
    )


class TimeMixerForRUL(nn.Module):
    """TimeMixer backbone adapted for scalar RUL regression.

    The TimeMixer backbone produces encoded representations
    ``[B(*N), seq_len, d_model]`` at the finest scale.  A global average
    pooling followed by a linear head projects to a scalar prediction
    per sample.

    Parameters
    ----------
    configs:
        SimpleNamespace with TimeMixer hyperparameters.
    c_in:
        Number of input features (sensor + operational columns).
    """

    def __init__(self, configs: types.SimpleNamespace, c_in: int):
        super().__init__()
        from timemixer_new.timemixer_layers.TimeMixer import Model as TimeMixerModel
        self.backbone = TimeMixerModel(configs)
        self.channel_independence = configs.channel_independence
        self.c_in = c_in
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len

        if self.channel_independence == 1:
            self.head = nn.Linear(configs.d_model, 1)
        else:
            self.head = nn.Linear(configs.d_model, 1)
        nn.init.xavier_uniform_(self.head.weight)
        self.dropout = nn.Dropout(configs.dropout)

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
        B, T, N = x.size()
        enc_out = self.backbone(x)  # [B*N, T, d_model] if CI=1, else [B, T, d_model]

        if self.channel_independence == 1:
            # enc_out is [B*N, T, d_model]; pool across time, then reshape
            pooled = enc_out.mean(dim=1)  # [B*N, d_model]
            pooled = pooled.reshape(B, N, self.d_model)  # [B, N, d_model]
            pooled = pooled.mean(dim=1)  # [B, d_model]
        else:
            pooled = enc_out.mean(dim=1)  # [B, d_model]

        pooled = F.gelu(pooled)
        pooled = self.dropout(pooled)
        return self.head(pooled)  # [B, 1]

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(
    cfg: Dict,
    c_in: int,
    seq_len: int,
    device: torch.device,
) -> TimeMixerForRUL:
    """Instantiate ``TimeMixerForRUL`` from a flat config dict and move to device.

    Parameters
    ----------
    cfg:
        Flat config dict containing TimeMixer hyperparameters.
    c_in:
        Number of input features.
    seq_len:
        Window / sequence length.
    device:
        Target torch device.
    """
    configs = _make_timemixer_configs(cfg, c_in, seq_len)
    return TimeMixerForRUL(configs, c_in).to(device)
