from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from features import load_raw_posts


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train a DistilBERT classifier to detect viral Hacker News posts."
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
        default=repo_root / "reports" / "transformer",
        help="Directory where checkpoints and metrics will be stored.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="If provided, limit the dataset to the first N rows.",
    )
    parser.add_argument(
        "--viral-threshold",
        type=int,
        default=250,
        help="Score threshold that defines a viral post.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.85,
        help="Fraction of earliest posts to allocate to training (chronological split).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="distilbert-base-uncased",
        help="Name of the Hugging Face model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=160,
        help="Maximum tokenized sequence length for title+URL pairs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Per-device batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Initial learning rate for AdamW.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay applied to AdamW.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=4,
        help="Total fine-tuning epochs.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of steps used for LR warmup (cosine schedule).",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Gradient accumulation steps to simulate larger batches.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed-precision training if supported.",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce memory footprint.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Number of evaluation steps without improvement before early stopping.",
    )
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clean_and_deduplicate(
    df: pd.DataFrame, viral_threshold: int
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    work = df.copy()
    work["title"] = work["title"].fillna("").astype(str).str.strip()
    work["url"] = work["url"].fillna("").astype(str).str.strip()
    work["id"] = work["id"].astype(str)

    total_before = len(work)
    non_null_urls = (work["url"] != "").sum()

    dedup_key = np.where(
        work["url"] != "",
        work["url"],
        "self-post://" + work["id"],
    )
    work["_url_key"] = dedup_key
    keep_idx = work.groupby("_url_key", sort=False)["score"].idxmax()
    work = work.loc[keep_idx].copy()
    work = work.sort_values("time").reset_index(drop=True)
    work["label"] = (work["score"] > viral_threshold).astype(np.float32)
    work["url"] = work["url"].replace("", "[self-post]")

    stats = {
        "rows_before": int(total_before),
        "rows_after": int(len(work)),
        "deduplicated": int(total_before - len(work)),
        "non_null_urls_before": int(non_null_urls),
    }
    return work.drop(columns="_url_key"), stats


def chronological_split(
    df: pd.DataFrame, train_fraction: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    order = df.sort_values("time").reset_index(drop=True)
    split_idx = int(len(order) * train_fraction)
    train_df = order.iloc[:split_idx].reset_index(drop=True)
    valid_df = order.iloc[split_idx:].reset_index(drop=True)
    return train_df, valid_df


def compute_pos_weight(labels: Iterable[float]) -> float:
    labels_arr = np.asarray(labels, dtype=np.float32)
    pos = labels_arr.sum()
    neg = len(labels_arr) - pos
    if pos == 0:
        return 1.0
    return float(neg / pos)


def find_best_threshold(
    labels: np.ndarray, probs: np.ndarray
) -> Tuple[float, Dict[str, float]]:
    if len(labels) == 0:
        return 0.5, {}
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    if len(thresholds) == 0:
        return 0.5, {}
    precision = precision[:-1]
    recall = recall[:-1]
    thresholds = thresholds
    denom = precision + recall
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    f1 = np.nan_to_num(f1, nan=0.0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), {
        "threshold_precision": float(precision[best_idx]),
        "threshold_recall": float(recall[best_idx]),
        "threshold_f1": float(f1[best_idx]),
    }


def _safe_metric(fn, *args) -> float:
    try:
        return float(fn(*args))
    except ValueError:
        return 0.0


class HNPostDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        record = self.frame.iloc[idx]
        paired_url = record["url"] if isinstance(record["url"], str) else ""
        tokens = self.tokenizer(
            record["title"],
            None if paired_url in {"", "[self-post]"} else paired_url,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        tokens["labels"] = np.array(record["label"], dtype=np.float32)
        return tokens


class ViralTrainer(Trainer):
    def __init__(self, pos_weight: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ):
        labels = inputs.pop("labels").float()
        outputs = model(**inputs)
        logits = outputs.logits.view(-1)
        loss_fct = torch.nn.BCEWithLogitsLoss(
            pos_weight=self._pos_weight.to(logits.device)
        )
        loss = loss_fct(logits, labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def build_metrics_fn():
    def _metrics(eval_pred):
        logits, labels = eval_pred
        probs = sigmoid(np.array(logits).reshape(-1))
        labels = np.array(labels).reshape(-1)
        threshold, stats = find_best_threshold(labels, probs)
        preds = (probs >= threshold).astype(int)
        metrics = {
            "roc_auc": _safe_metric(roc_auc_score, labels, probs),
            "pr_auc": _safe_metric(average_precision_score, labels, probs),
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": _safe_metric(f1_score, labels, preds),
            "best_threshold": float(threshold),
        }
        metrics.update(stats)
        return metrics

    return _metrics


def summarize_final_metrics(
    labels: np.ndarray, probs: np.ndarray
) -> Dict[str, float]:
    threshold, stats = find_best_threshold(labels, probs)
    preds = (probs >= threshold).astype(int)
    metrics = {
        "roc_auc": _safe_metric(roc_auc_score, labels, probs),
        "pr_auc": _safe_metric(average_precision_score, labels, probs),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": _safe_metric(f1_score, labels, preds),
        "decision_threshold": float(threshold),
    }
    metrics.update(stats)
    return metrics


@dataclass
class TrainingArtifacts:
    metrics: Dict[str, float]
    data_stats: Dict[str, int]
    pos_weight: float


def train(args: argparse.Namespace) -> TrainingArtifacts:
    set_seed(args.seed)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_posts(args.data_path, n_rows=args.n_rows)
    cleaned, data_stats = clean_and_deduplicate(raw, args.viral_threshold)
    print(
        f"Loaded {len(raw):,} rows → {len(cleaned):,} after URL dedup "
        f"(removed {data_stats['deduplicated']:,})."
    )
    baseline = cleaned["label"].mean()
    print(f"Class balance (viral proportion): {baseline:.2%}")

    train_df, valid_df = chronological_split(cleaned, args.train_fraction)
    print(
        f"Chronological split sizes — train: {len(train_df):,}, "
        f"valid: {len(valid_df):,}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        problem_type="single_label_classification",
    )

    train_dataset = HNPostDataset(train_df, tokenizer, args.max_length)
    valid_dataset = HNPostDataset(valid_df, tokenizer, args.max_length)

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None,
    )
    pos_weight = compute_pos_weight(train_df["label"].values)
    print(f"Positive class weight: {pos_weight:.2f}")

    training_args = TrainingArguments(
        output_dir=str(args.reports_dir / "checkpoints"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="pr_auc",
        load_best_model_at_end=True,
        greater_is_better=True,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=50,
        report_to=[],
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        seed=args.seed,
    )

    trainer = ViralTrainer(
        pos_weight=pos_weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=build_metrics_fn(),
        callbacks=(
            [
                EarlyStoppingCallback(
                    early_stopping_patience=args.patience,
                    early_stopping_threshold=0.0,
                )
            ]
            if args.patience and args.patience > 0
            else None
        ),
    )

    trainer.train()
    predictions = trainer.predict(valid_dataset)
    logits = predictions.predictions.reshape(-1)
    probs = sigmoid(logits)
    metrics = summarize_final_metrics(valid_df["label"].values, probs)

    artifacts = {
        "metrics": metrics,
        "data_stats": data_stats,
        "class_weight": pos_weight,
        "hyperparams": {
            "model_name": args.model_name,
            "max_length": args.max_length,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "train_fraction": args.train_fraction,
        },
    }

    metrics_path = args.reports_dir / "transformer_metrics.json"
    metrics_path.write_text(json.dumps(artifacts, indent=2))

    preds_path = args.reports_dir / "transformer_validation_predictions.csv"
    output = valid_df.copy()
    output["probability"] = probs
    output.to_csv(preds_path, index=False)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved validation predictions to {preds_path}")
    return TrainingArtifacts(metrics=metrics, data_stats=data_stats, pos_weight=pos_weight)


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

