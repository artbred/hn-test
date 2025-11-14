from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from embeddings import compute_sentence_embeddings
from features import load_raw_posts


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train a text-only virality classifier using Qwen3 embeddings."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=repo_root / "data" / "hn_posts.csv",
        help="Absolute path to the hn_posts.csv file.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=repo_root / "reports",
        help="Directory where metrics and artifacts will be stored.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Fraction of earliest posts to use for training (chronological split).",
    )
    parser.add_argument(
        "--viral-threshold",
        type=int,
        default=500,
        help="Score threshold that defines a viral post.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="Qwen/Qwen3-Embedding-0.6B",
        help="SentenceTransformer-compatible embedding model name.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=128,
        help="Batch size for embedding inference.",
    )
    parser.add_argument(
        "--embedding-max-seq-length",
        type=int,
        default=1024,
        help="Maximum token length fed into the embedding model (truncates longer samples).",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=repo_root / "data" / "embeddings" / "text_qwen3.npy",
        help="Path to cache text embeddings (.npy).",
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="Normalize embeddings to unit length before training.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for the classification head.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout applied within the classification head.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Training epochs for the classification head.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for classifier fine-tuning.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for Adam optimizer.",
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=("f1", "precision_at_recall"),
        default="f1",
        help="How to pick the probability threshold on validation data.",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.3,
        help="Recall constraint used by precision_at_recall strategy.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Whether to enable trust_remote_code when loading the embedding model.",
    )
    return parser.parse_args()


def build_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    work = df[["by", "time", "is_viral", "score"]].copy()
    work = work.sort_values("time")
    grouped = work.groupby("by", sort=False, dropna=False)
    prior_posts = grouped.cumcount().astype(np.float32)
    prior_viral = (grouped["is_viral"].cumsum() - work["is_viral"]).astype(np.float32)
    prior_scores = (grouped["score"].cumsum() - work["score"]).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_score = prior_scores / prior_posts.replace(0, np.nan)
        viral_rate = prior_viral / prior_posts.replace(0, np.nan)
    hours_since_last = grouped["time"].diff() / 3600.0
    stats = pd.DataFrame(
        {
            "user_prior_posts": prior_posts,
            "user_prior_viral": prior_viral,
            "user_prior_mean_score": mean_score.astype(np.float32),
            "user_prior_viral_rate": viral_rate.astype(np.float32),
            "user_hours_since_last": hours_since_last.astype(np.float32),
        },
        index=work.index,
    )
    stats = stats.reindex(df.index)
    stats["user_prior_posts"] = stats["user_prior_posts"].fillna(0.0)
    stats["user_prior_viral"] = stats["user_prior_viral"].fillna(0.0)
    stats["user_prior_mean_score"] = stats["user_prior_mean_score"].fillna(
        df["score"].mean()
    )
    stats["user_prior_viral_rate"] = stats["user_prior_viral_rate"].fillna(
        df["is_viral"].mean()
    )
    median_hours = stats["user_hours_since_last"].median()
    stats["user_hours_since_last"] = stats["user_hours_since_last"].fillna(
        24.0 if np.isnan(median_hours) else float(median_hours)
    )
    return stats


def prepare_formatted_text(
    df: pd.DataFrame, user_stats: pd.DataFrame | None = None
) -> List[str]:
    titles = df["title"].fillna("").astype(str)
    users = df["by"].fillna("unknown").astype(str)
    urls = df["url"].fillna("").astype(str)
    texts = df.get("text", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    stats_records = None
    if user_stats is not None:
        stats_records = user_stats.reindex(df.index).to_dict("records")
    formatted = []
    for idx, (title, user, url, text) in enumerate(
        zip(titles, users, urls, texts, strict=False)
    ):
        lines = [
            f"Title: {title.strip()}",
            f"User: {user.strip()}",
            f"Url: {url.strip()}",
        ]
        if text:
            lines.append(f"Text: {text.strip()}")
        if stats_records is not None:
            record = stats_records[idx]
            lines.append(
                "UserStats: "
                f"prior_posts={int(record['user_prior_posts'])}; "
                f"prior_viral={int(record['user_prior_viral'])}; "
                f"viral_rate={record['user_prior_viral_rate']:.2f}; "
                f"mean_score={record['user_prior_mean_score']:.1f}; "
                f"hours_since_last={record['user_hours_since_last']:.1f}"
            )
        formatted.append("\n".join(lines))
    return formatted


def chronological_split(
    embeddings: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    train_fraction: float,
    ids: np.ndarray | pd.Series | None = None,
) -> Dict[str, np.ndarray]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    order = np.argsort(timestamps)
    n_train = int(len(order) * train_fraction)
    idx_train = order[:n_train]
    idx_valid = order[n_train:]
    splits = {
        "X_train": embeddings[idx_train],
        "y_train": labels[idx_train],
        "X_valid": embeddings[idx_valid],
        "y_valid": labels[idx_valid],
    }
    if ids is not None:
        ids_array = np.asarray(ids)
        splits["ids_valid"] = ids_array[idx_valid]
    return splits


class TextHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        hidden_out = max(hidden_dim // 2, 32)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_out, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class TrainerConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    dropout: float


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_text_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    config: TrainerConfig,
) -> Tuple[TextHead, np.ndarray]:
    device = resolve_device()
    model = TextHead(
        input_dim=X_train.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)
    pos_rate = y_train.mean()
    if pos_rate <= 0 or pos_rate >= 1:
        pos_weight = None
    else:
        pos_weight = torch.tensor([(1 - pos_rate) / pos_rate], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float(),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model.train()
    for _ in range(config.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        valid_tensor = torch.from_numpy(X_valid).float().to(device)
        logits = model(valid_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()
    return model, probs


def evaluate_metrics(
    y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, preds)),
        "accuracy": float(accuracy_score(y_true, preds)),
    }


def tune_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    strategy: str,
    target_recall: float,
) -> Tuple[float, Dict[str, float]]:
    if proba.size == 0:
        return 0.5, {}
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    if thresholds.size == 0:
        return 0.5, {}
    precision = precision[:-1]
    recall = recall[:-1]
    if strategy == "f1":
        denom = precision + recall
        with np.errstate(invalid="ignore", divide="ignore"):
            f1 = np.where(denom > 0, 2 * (precision * recall) / denom, 0.0)
        f1 = np.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)
        best_idx = int(np.nanargmax(f1))
        return thresholds[best_idx], {
            "threshold_precision": float(precision[best_idx]),
            "threshold_recall": float(recall[best_idx]),
            "threshold_f1": float(f1[best_idx]),
        }
    mask = recall >= target_recall
    if not mask.any():
        best_idx = int(np.nanargmax(recall))
    else:
        candidate_idx = np.arange(len(thresholds))[mask]
        best_idx = int(candidate_idx[np.argmax(precision[mask])])
    return thresholds[best_idx], {
        "threshold_precision": float(precision[best_idx]),
        "threshold_recall": float(recall[best_idx]),
    }


def summarize_model_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    strategy: str,
    target_recall: float,
) -> Tuple[float, Dict[str, float]]:
    threshold, stats = tune_threshold(
        y_true,
        probs,
        strategy=strategy,
        target_recall=target_recall,
    )
    metrics = evaluate_metrics(y_true, probs, threshold=threshold)
    metrics["decision_threshold"] = float(threshold)
    metrics["threshold_strategy"] = strategy
    if strategy == "precision_at_recall":
        metrics["target_recall"] = float(target_recall)
    metrics.update(stats)
    return threshold, metrics


def save_metrics(path: Path, metrics: Dict[str, float]) -> None:
    path.write_text(json.dumps(metrics, indent=2))


def save_predictions(
    path: Path,
    ids: np.ndarray,
    probs: np.ndarray,
) -> None:
    frame = pd.DataFrame({"id": ids, "probability": probs})
    frame.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    raw = load_raw_posts(args.data_path)
    raw["title"] = raw["title"].fillna("")
    raw["by"] = raw["by"].fillna("unknown")
    raw["url"] = raw["url"].fillna("")
    raw["text"] = raw.get(
        "text",
        pd.Series([""] * len(raw), index=raw.index),
    ).fillna("")
    raw["is_viral"] = (raw["score"] > args.viral_threshold).astype(np.int32)
    user_stats = build_user_stats(raw)
    formatted_inputs = prepare_formatted_text(raw, user_stats=user_stats)
    embeddings = compute_sentence_embeddings(
        formatted_inputs,
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
        cache_path=args.embedding_cache,
        normalize=args.normalize_embeddings,
        trust_remote_code=args.trust_remote_code,
        max_seq_length=args.embedding_max_seq_length,
    ).astype(np.float32)
    labels = raw["is_viral"].to_numpy(dtype=np.float32)
    timestamps = raw["time"].to_numpy(dtype=np.int64)
    splits = chronological_split(
        embeddings,
        labels,
        timestamps,
        args.train_fraction,
        ids=raw["id"].to_numpy(),
    )
    config = TrainerConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model, valid_probs = train_text_head(
        splits["X_train"],
        splits["y_train"],
        splits["X_valid"],
        config=config,
    )
    _, metrics = summarize_model_metrics(
        splits["y_valid"],
        valid_probs,
        strategy=args.threshold_strategy,
        target_recall=args.target_recall,
    )
    metrics_path = args.reports_dir / "text_model_metrics.json"
    preds_path = args.reports_dir / "text_model_validation_predictions.csv"
    model_path = args.reports_dir / "text_model_head.pt"
    save_metrics(metrics_path, metrics)
    save_predictions(preds_path, splits.get("ids_valid", np.arange(len(valid_probs))), valid_probs)
    torch.save(model.state_dict(), model_path)
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved validation predictions to {preds_path}")
    print(f"Saved classifier weights to {model_path}")


if __name__ == "__main__":
    main()


