from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .model import build_model


class RULDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    delta = pred - true
    return float(np.sum(np.where(delta < 0, np.exp(-delta / 13.0) - 1.0, np.exp(delta / 10.0) - 1.0)))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def selection_value(metric_name: str, value_rmse: float, value_score: float, score_weight: float) -> float:
    if metric_name == "rmse":
        return float(value_rmse)
    if metric_name == "score":
        return float(value_score)
    if metric_name == "hybrid":
        return float(value_rmse + score_weight * value_score)
    raise ValueError(metric_name)


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, max_rul: float, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        preds.append(np.clip(model(xb.to(device)).cpu().numpy().reshape(-1), 0.0, max_rul))
        trues.append(yb.numpy().reshape(-1))
    return np.concatenate(preds), np.concatenate(trues)


@torch.no_grad()
def predict_array(model: nn.Module, x: np.ndarray, max_rul: float, device: torch.device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    preds = []
    for idx in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[idx : idx + batch_size]).to(device)
        preds.append(model(xb).cpu().numpy().reshape(-1))
    return np.clip(np.concatenate(preds), 0.0, max_rul)


def train_model(
    cfg: Dict[str, object],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int,
    patience: int,
    selection_metric: str,
    score_weight: float,
    device: torch.device,
) -> Tuple[nn.Module, float, float, int, float]:
    model = build_model(cfg, c_in=x_train.shape[2], seq_len=x_train.shape[1], device=device)
    train_loader = DataLoader(RULDataset(x_train, y_train), batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(RULDataset(x_val, y_val), batch_size=int(cfg["batch_size"]) * 4, shuffle=False, num_workers=0)

    optimizer = torch.optim.RAdam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=float(cfg["lr"]) * 0.05)
    loss_fn = nn.HuberLoss(delta=float(cfg["huber_delta"]))

    best_state = None
    best_rmse = float("inf")
    best_score = float("inf")
    best_metric = float("inf")
    best_epoch = 0
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        pred, true = predict_loader(model, val_loader, max_rul=float(cfg["max_rul"]), device=device)
        val_rmse = rmse(pred, true)
        val_score = nasa_score(pred, true)
        val_metric = selection_value(selection_metric, val_rmse, val_score, score_weight)
        if val_metric < best_metric - 1e-6:
            best_metric = val_metric
            best_rmse = val_rmse
            best_score = val_score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_rmse, best_score, best_epoch, best_metric


def fit_fixed_epochs(cfg: Dict[str, object], x_train: np.ndarray, y_train: np.ndarray, epochs: int, device: torch.device) -> nn.Module:
    model = build_model(cfg, c_in=x_train.shape[2], seq_len=x_train.shape[1], device=device)
    train_loader = DataLoader(RULDataset(x_train, y_train), batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=0, drop_last=True)
    optimizer = torch.optim.RAdam(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=float(cfg["lr"]) * 0.05)
    loss_fn = nn.HuberLoss(delta=float(cfg["huber_delta"]))
    for _ in range(max(1, epochs)):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()
    return model
