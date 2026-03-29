from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

import mambasl_new.mamba_layers.MambaBlock as mamba_block_mod
from mambasl_new.mamba_layers.Embed import PositionalEmbedding
from mambasl_new.mamba_layers.MambaBlock import Mamba_TimeVariant


def configure_device(prefer_gpu: bool = True) -> torch.device:
    device = torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        mamba_block_mod.selective_scan_fn = selective_scan_ref
    return device


class ConvTokenEmbed(nn.Module):
    def __init__(self, c_in: int, d_model: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(
            c_in,
            d_model,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="replicate",
            bias=False,
        )
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x):
        return self.conv(x.permute(0, 2, 1)).transpose(1, 2)


class MambaSLRUL(nn.Module):
    def __init__(
        self,
        c_in: int,
        d_model: int,
        d_state: int,
        d_conv: int,
        expand: int,
        num_kernels: int,
        seq_len: int,
        tv_dt: bool,
        tv_B: bool,
        tv_C: bool,
        use_D: bool,
        projection: str,
        dropout: float,
        device: torch.device,
    ):
        super().__init__()
        self.projection = projection
        if num_kernels > 0:
            self.token_emb = ConvTokenEmbed(c_in, d_model, kernel_size=num_kernels)
            self.pos_emb = PositionalEmbedding(d_model=d_model, max_len=max(5000, seq_len))
            self.drop_emb = nn.Dropout(dropout)
            d_input = d_model
        else:
            self.token_emb = None
            self.pos_emb = None
            self.drop_emb = None
            d_input = c_in

        self.mamba = nn.Sequential(
            Mamba_TimeVariant(
                d_model=d_model,
                d_input=d_input,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                timevariant_dt=tv_dt,
                timevariant_B=tv_B,
                timevariant_C=tv_C,
                use_D=use_D,
                use_fast_path=False,
                device=device,
            ),
            nn.LayerNorm(d_model),
            nn.SiLU(),
        )
        in_feats = d_model * seq_len if projection == "full" else d_model
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feats, 1))
        nn.init.xavier_uniform_(self.head[-1].weight)

    def forward(self, x):
        if self.token_emb is not None:
            x = self.token_emb(x) + self.pos_emb(x)
            x = self.drop_emb(x)
        hidden = self.mamba(x)
        if self.projection == "last":
            pooled = hidden[:, -1, :]
        elif self.projection == "avg":
            pooled = hidden.mean(dim=1)
        else:
            pooled = hidden.reshape(hidden.size(0), -1)
        return self.head(pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: Dict[str, object], c_in: int, seq_len: int, device: torch.device) -> MambaSLRUL:
    return MambaSLRUL(
        c_in=c_in,
        d_model=int(cfg["d_model"]),
        d_state=int(cfg["d_state"]),
        d_conv=int(cfg["d_conv"]),
        expand=int(cfg["expand"]),
        num_kernels=int(cfg["num_kernels"]),
        seq_len=seq_len,
        tv_dt=bool(cfg["tv_dt"]),
        tv_B=bool(cfg["tv_B"]),
        tv_C=bool(cfg["tv_C"]),
        use_D=bool(cfg["use_D"]),
        projection=str(cfg["projection"]),
        dropout=float(cfg["dropout"]),
        device=device,
    ).to(device)
