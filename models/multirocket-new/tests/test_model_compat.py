import numpy as np
import torch

from multirocket_new.model import MRHySPRegressor, _HydraSparseScaler, _to_numpy


def test_hydra_sparse_scaler_matches_reference_formula():
    x = torch.tensor([[0.0, 1.0, 4.0], [9.0, 0.0, 16.0]], dtype=torch.float32)
    scaler = _HydraSparseScaler()
    out = scaler.fit_transform(x)

    sqrt_x = x.clamp(0).sqrt()
    epsilon = (sqrt_x == 0).float().mean(0) ** 4 + 1e-8
    expected = ((sqrt_x - sqrt_x.mean(0)) * (sqrt_x != 0)) / (sqrt_x.std(0) + epsilon)
    assert torch.allclose(out, expected)


def test_to_numpy_handles_tensor_and_array():
    arr = np.array([[1.0, 2.0]], dtype=np.float32)
    assert np.array_equal(_to_numpy(arr), arr)

    tensor = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    out = _to_numpy(tensor)
    assert isinstance(out, np.ndarray)
    assert out.shape == (1, 2)


def test_mrhysp_regressor_fit_predict_smoke():
    x = np.random.RandomState(0).randn(6, 2, 16).astype(np.float32)
    y = np.linspace(0.0, 1.0, num=6, dtype=np.float32)

    model = MRHySPRegressor(
        mr_num_kernels=84,
        n_kernels=2,
        n_groups=2,
        n_kernels_sp=8,
        n_jobs=1,
        random_state=0,
    )
    preds = model.fit(x, y).predict(x)
    assert preds.shape == (6,)
