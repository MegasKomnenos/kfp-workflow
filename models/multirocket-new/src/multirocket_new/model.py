"""SPROCKET transformer and MR-HY-SP regressor ported from the reference repo."""

from __future__ import annotations

import numpy as np
from aeon.regression import BaseRegressor
from aeon.regression.convolution_based._hydra import _SparseScaler
from aeon.transformations.collection.convolution_based import MultiRocketMultivariate
from aeon.transformations.collection.convolution_based._hydra import HydraTransformer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from typing import Optional


def _generate_kernels(
    n_timepoints: int,
    n_kernels: int,
    n_channels: int,
    rng: np.random.RandomState,
):
    lengths = rng.choice(np.array([7, 9, 11], dtype=np.int32), n_kernels).astype(np.int32)
    num_ci = np.zeros(n_kernels, dtype=np.int32)
    for index in range(n_kernels):
        limit = min(n_channels, int(lengths[index]))
        num_ci[index] = max(1, int(2 ** rng.uniform(0.0, np.log2(limit + 1))))

    n_w_total = int(np.dot(lengths.astype(np.int64), num_ci.astype(np.int64)))
    n_c_total = int(num_ci.sum())
    weights = np.zeros(n_w_total, dtype=np.float32)
    biases = np.zeros(n_kernels, dtype=np.float32)
    dilations = np.zeros(n_kernels, dtype=np.int32)
    channel_indices = np.zeros(n_c_total, dtype=np.int32)
    w_starts = np.zeros(n_kernels, dtype=np.int32)
    w_ends = np.zeros(n_kernels, dtype=np.int32)
    c_starts = np.zeros(n_kernels, dtype=np.int32)
    c_ends = np.zeros(n_kernels, dtype=np.int32)

    w_off = 0
    c_off = 0
    for index in range(n_kernels):
        length = int(lengths[index])
        nch = int(num_ci[index])
        weight_block = rng.normal(0.0, 1.0, nch * length).astype(np.float32)
        for channel in range(nch):
            start = channel * length
            weight_block[start:start + length] -= weight_block[start:start + length].mean()
        weights[w_off:w_off + nch * length] = weight_block
        w_starts[index], w_ends[index] = w_off, w_off + nch * length
        w_off += nch * length

        chosen = rng.choice(n_channels, nch, replace=False).astype(np.int32)
        channel_indices[c_off:c_off + nch] = chosen
        c_starts[index], c_ends[index] = c_off, c_off + nch
        c_off += nch

        biases[index] = rng.uniform(-1.0, 1.0)
        max_exp = np.log2(max(1.0, (n_timepoints - 1) / max(1, length - 1)))
        dilations[index] = max(1, int(2 ** rng.uniform(0.0, max_exp)))

    return (
        lengths, weights, biases, dilations, num_ci,
        channel_indices, w_starts, w_ends, c_starts, c_ends,
    )


def _conv_batch(x_batch: np.ndarray, kw: np.ndarray, bias: float, dilation: int) -> np.ndarray:
    n_cases, n_t = x_batch.shape
    length = len(kw)
    mid = length // 2
    out = np.full((n_cases, n_t), bias, dtype=np.float32)
    for index in range(length):
        offset = (index - mid) * dilation
        dst_s = max(0, -offset)
        dst_e = min(n_t, n_t - offset)
        if dst_s >= dst_e:
            continue
        src_s = dst_s + offset
        src_e = dst_e + offset
        out[:, dst_s:dst_e] += kw[index] * x_batch[:, src_s:src_e]
    return out


def _activations_for_kernel(
    x: np.ndarray,
    kernel_weights_2d: np.ndarray,
    bias: float,
    dilation: int,
    chan_idx: np.ndarray,
) -> np.ndarray:
    n_cases, _, n_t = x.shape
    n_cu = len(chan_idx)
    act = np.empty((n_cases, n_cu, n_t), dtype=np.float32)
    for index in range(n_cu):
        act[:, index, :] = _conv_batch(x[:, chan_idx[index], :], kernel_weights_2d[index], bias, dilation)
    return act


class SPRocketTransformer:
    def __init__(
        self,
        n_kernels: int = 512,
        proto_per_kernel: float = 4.0,
        dist_id: str = "euclidean",
        random_state: Optional[int] = None,
    ):
        self.n_kernels = n_kernels
        self.proto_per_kernel = proto_per_kernel
        self.dist_id = dist_id
        self.random_state = random_state

    def fit(self, x: np.ndarray, y=None) -> "SPRocketTransformer":
        x = np.asarray(x, dtype=np.float32)
        n_cases, n_channels, n_timepoints = x.shape
        rng = np.random.RandomState(self.random_state)
        self._kernels = _generate_kernels(n_timepoints, self.n_kernels, n_channels, rng)
        lengths, weights, biases, dilations, num_ci, chan_idx, w_starts, w_ends, c_starts, c_ends = self._kernels
        self._n_protos = 1 + int(np.log(n_cases) / np.log(self.proto_per_kernel))
        self._kernel_points: list[np.ndarray] = []
        for kernel_index in range(self.n_kernels):
            length = int(lengths[kernel_index])
            nch = int(num_ci[kernel_index])
            kw_2d = weights[w_starts[kernel_index]:w_ends[kernel_index]].reshape(nch, length)
            ci = chan_idx[c_starts[kernel_index]:c_ends[kernel_index]]
            proto_idx = rng.choice(n_cases, size=self._n_protos, replace=False)
            x_proto = x[proto_idx]
            proto_act = _activations_for_kernel(
                x_proto, kw_2d, float(biases[kernel_index]), int(dilations[kernel_index]), ci
            )
            self._kernel_points.append(proto_act)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        n_cases = x.shape[0]
        n_protos = self._n_protos
        lengths, weights, biases, dilations, num_ci, chan_idx, w_starts, w_ends, c_starts, c_ends = self._kernels
        features = np.empty((n_cases, self.n_kernels * n_protos), dtype=np.float32)
        for kernel_index in range(self.n_kernels):
            length = int(lengths[kernel_index])
            nch = int(num_ci[kernel_index])
            kw_2d = weights[w_starts[kernel_index]:w_ends[kernel_index]].reshape(nch, length)
            ci = chan_idx[c_starts[kernel_index]:c_ends[kernel_index]]
            protos = self._kernel_points[kernel_index]
            act = _activations_for_kernel(
                x, kw_2d, float(biases[kernel_index]), int(dilations[kernel_index]), ci
            )
            diff = act[:, np.newaxis, :, :] - protos[np.newaxis, :, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=(-2, -1)))
            col_start = kernel_index * n_protos
            features[:, col_start:col_start + n_protos] = dists
        return features


class MRHySPRegressor(BaseRegressor):
    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "algorithm_type": "convolution",
    }

    def __init__(
        self,
        mr_num_kernels: int = 6250,
        n_kernels: int = 8,
        n_groups: int = 64,
        n_kernels_sp: int = 512,
        n_jobs: int = 1,
        random_state: Optional[int] = None,
    ):
        self.mr_num_kernels = mr_num_kernels
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.n_kernels_sp = n_kernels_sp
        self.n_jobs = n_jobs
        self.random_state = random_state
        super().__init__()

    def _fit(self, x: np.ndarray, y: np.ndarray):
        self._hydra = HydraTransformer(
            n_kernels=self.n_kernels,
            n_groups=self.n_groups,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        xt_hydra = self._hydra.fit_transform(x)
        self._hydra_scaler = _SparseScaler()
        xt_hydra = self._hydra_scaler.fit_transform(xt_hydra).numpy()

        self._mr = MultiRocketMultivariate(
            num_kernels=self.mr_num_kernels,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        xt_mr = self._mr.fit_transform(x).values
        self._mr_scaler = StandardScaler()
        xt_mr = self._mr_scaler.fit_transform(xt_mr)

        self._sp = SPRocketTransformer(
            n_kernels=self.n_kernels_sp,
            random_state=self.random_state,
        )
        xt_sp = self._sp.fit(x, y).transform(x)

        xt = np.concatenate([xt_hydra, xt_mr, xt_sp], axis=1)
        self._ridge = RidgeCV(alphas=np.logspace(-3, 3, 10))
        self._ridge.fit(xt, y)
        return self

    def _predict(self, x: np.ndarray) -> np.ndarray:
        xt_hydra = self._hydra_scaler.transform(self._hydra.transform(x)).numpy()
        xt_mr = self._mr_scaler.transform(self._mr.transform(x).values)
        xt_sp = self._sp.transform(x)
        return self._ridge.predict(np.concatenate([xt_hydra, xt_mr, xt_sp], axis=1))
