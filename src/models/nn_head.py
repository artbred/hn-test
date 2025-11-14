from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class NeuralHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class NeuralHeadConfig:
    epochs: int = 25
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4


def _make_loader(
    features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features).float(),
        torch.from_numpy(labels).float(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_neural_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    config: NeuralHeadConfig | None = None,
) -> Tuple[NeuralHead, np.ndarray]:
    if config is None:
        config = NeuralHeadConfig()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = NeuralHead(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_loader = _make_loader(X_train, y_train, config.batch_size, shuffle=True)
    valid_tensor = torch.from_numpy(X_valid).float().to(device)

    model.train()
    for _ in range(config.epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(valid_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return model, probs

