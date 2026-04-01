"""TimeMixer model architecture.

Adapted from the original TimeMixer implementation (ICLR 2024).
This module contains the core multi-scale mixing blocks (PastDecomposableMixing)
and the full Model class.
"""

import torch
import torch.nn as nn

from .Autoformer_EncDec import series_decomp
from .Embed import DataEmbedding_wo_pos
from .StandardNorm import Normalize


class DFT_series_decomp(nn.Module):
    """FFT-based series decomposition: retains top-k frequency components
    as the seasonal part, remainder as trend."""

    def __init__(self, top_k=5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, self.top_k)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend


class MultiScaleSeasonMixing(nn.Module):
    """Bottom-up mixing of seasonal patterns across scales."""

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.down_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** (i + 1)),
                    ),
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, season_list):
        out_high = season_list[0]
        out_low = season_list[1]
        out_season_list = [out_high.permute(0, 2, 1)]

        for i in range(len(season_list) - 1):
            out_low_res = self.down_sampling_layers[i](out_high)
            out_low = out_low + out_low_res
            out_high = out_low
            if i + 2 <= len(season_list) - 1:
                out_low = season_list[i + 2]
            out_season_list.append(out_high.permute(0, 2, 1))

        return out_season_list


class MultiScaleTrendMixing(nn.Module):
    """Top-down mixing of trend patterns across scales."""

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.up_sampling_layers = torch.nn.ModuleList(
            [
                nn.Sequential(
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** (i + 1)),
                        seq_len // (down_sampling_window ** i),
                    ),
                    nn.GELU(),
                    torch.nn.Linear(
                        seq_len // (down_sampling_window ** i),
                        seq_len // (down_sampling_window ** i),
                    ),
                )
                for i in reversed(range(down_sampling_layers))
            ]
        )

    def forward(self, trend_list):
        trend_list_reverse = trend_list.copy()
        trend_list_reverse.reverse()
        out_low = trend_list_reverse[0]
        out_high = trend_list_reverse[1]
        out_trend_list = [out_low.permute(0, 2, 1)]

        for i in range(len(trend_list_reverse) - 1):
            out_high_res = self.up_sampling_layers[i](out_low)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(trend_list_reverse) - 1:
                out_high = trend_list_reverse[i + 2]
            out_trend_list.append(out_low.permute(0, 2, 1))

        out_trend_list.reverse()
        return out_trend_list


class PastDecomposableMixing(nn.Module):
    """Core TimeMixer block: decompose → multi-scale season/trend mixing → recombine."""

    def __init__(
        self,
        seq_len,
        d_model,
        d_ff,
        down_sampling_window,
        down_sampling_layers,
        dropout,
        channel_independence,
        decomp_method="moving_avg",
        moving_avg=25,
        top_k=5,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.channel_independence = channel_independence

        if decomp_method == 'moving_avg':
            self.decompsition = series_decomp(moving_avg)
        elif decomp_method == "dft_decomp":
            self.decompsition = DFT_series_decomp(top_k)
        else:
            raise ValueError(f'Unknown decomposition method: {decomp_method}')

        if channel_independence == 0:
            self.cross_layer = nn.Sequential(
                nn.Linear(in_features=d_model, out_features=d_ff),
                nn.GELU(),
                nn.Linear(in_features=d_ff, out_features=d_model),
            )

        self.mixing_multi_scale_season = MultiScaleSeasonMixing(
            seq_len, down_sampling_window, down_sampling_layers,
        )
        self.mixing_multi_scale_trend = MultiScaleTrendMixing(
            seq_len, down_sampling_window, down_sampling_layers,
        )

        self.out_cross_layer = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=d_ff),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model),
        )

    def forward(self, x_list):
        length_list = []
        for x in x_list:
            _, T, _ = x.size()
            length_list.append(T)

        season_list = []
        trend_list = []
        for x in x_list:
            season, trend = self.decompsition(x)
            if self.channel_independence == 0:
                season = self.cross_layer(season)
                trend = self.cross_layer(trend)
            season_list.append(season.permute(0, 2, 1))
            trend_list.append(trend.permute(0, 2, 1))

        out_season_list = self.mixing_multi_scale_season(season_list)
        out_trend_list = self.mixing_multi_scale_trend(trend_list)

        out_list = []
        for ori, out_season, out_trend, length in zip(
            x_list, out_season_list, out_trend_list, length_list
        ):
            out = out_season + out_trend
            if self.channel_independence:
                out = ori + self.out_cross_layer(out)
            out_list.append(out[:, :length, :])
        return out_list


class Model(nn.Module):
    """TimeMixer model supporting multiple task types.

    For C-MAPSS RUL prediction, only the ``classification``-style pathway
    is used (encode → flatten → linear head), adapted by the wrapper in
    ``cmapss/model.py``.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.down_sampling_layers = configs.down_sampling_layers
        self.down_sampling_method = configs.down_sampling_method
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.use_norm = configs.use_norm

        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(
                seq_len=configs.seq_len,
                d_model=configs.d_model,
                d_ff=configs.d_ff,
                down_sampling_window=configs.down_sampling_window,
                down_sampling_layers=configs.down_sampling_layers,
                dropout=configs.dropout,
                channel_independence=configs.channel_independence,
                decomp_method=configs.decomp_method,
                moving_avg=configs.moving_avg,
                top_k=configs.top_k,
            )
            for _ in range(configs.e_layers)
        ])

        self.preprocess = series_decomp(configs.moving_avg)

        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.dropout)

        self.normalize_layers = torch.nn.ModuleList(
            [
                Normalize(configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for _ in range(configs.down_sampling_layers + 1)
            ]
        )

    def pre_enc(self, x_list):
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc):
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        elif self.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(
                in_channels=self.enc_in,
                out_channels=self.enc_in,
                kernel_size=3,
                padding=padding,
                stride=self.down_sampling_window,
                padding_mode='circular',
                bias=False,
            )
        else:
            return [x_enc]

        x_enc = x_enc.permute(0, 2, 1)  # B,T,C -> B,C,T
        x_enc_ori = x_enc
        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]

        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

        return x_enc_sampling_list

    def encode(self, x_enc):
        """Encode input through multi-scale decomposable mixing blocks.

        Returns the list of encoded representations at each scale.
        """
        x_enc_list = self.__multi_scale_process_inputs(x_enc)

        x_list = []
        for i, x in enumerate(x_enc_list):
            B, T, N = x.size()
            if self.use_norm:
                x = self.normalize_layers[i](x, 'norm')
            if self.channel_independence == 1:
                x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        enc_out_list = []
        x_list = self.pre_enc(x_list)
        for i, x in enumerate(x_list[0]):
            enc_out = self.enc_embedding(x, None)
            enc_out_list.append(enc_out)

        for i in range(len(self.pdm_blocks)):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        return enc_out_list

    def forward(self, x_enc):
        """Forward pass returning encoded representations at the finest scale."""
        enc_out_list = self.encode(x_enc)
        return enc_out_list[0]
