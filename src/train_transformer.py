from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas.api.types import is_categorical_dtype, is_object_dtype
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from features import FeatureEngineer, load_raw_posts


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
        "--tabular-hidden-dim",
        type=int,
        default=128,
        help="Hidden size for the tabular projection before fusion.",
    )
    parser.add_argument(
        "--max-cat-embed-dim",
        type=int,
        default=32,
        help="Maximum embedding dimension for categorical features.",
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
) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    order = df.sort_values("time").reset_index(drop=True)
    split_idx = int(len(order) * train_fraction)
    train_df = order.iloc[:split_idx].reset_index(drop=True)
    valid_df = order.iloc[split_idx:].reset_index(drop=True)
    return train_df, valid_df, split_idx


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


@dataclass
class StructuredFeatureMatrices:
    numeric_cols: List[str]
    categorical_cols: List[str]
    numeric_train: np.ndarray
    numeric_valid: np.ndarray
    categorical_train: Dict[str, np.ndarray]
    categorical_valid: Dict[str, np.ndarray]
    cat_vocab_sizes: Dict[str, int]
    numeric_scaler_mean: np.ndarray
    numeric_scaler_scale: np.ndarray


def prepare_structured_features(
    df: pd.DataFrame,
    split_idx: int,
    viral_threshold: int,
) -> StructuredFeatureMatrices:
    engineer = FeatureEngineer(
        viral_threshold=viral_threshold,
        title_embedding_model=None,
        title_embedding_dim=None,
        title_embedding_scale=False,
    )
    bundle = engineer.transform(df)
    features = bundle.features.reset_index(drop=True)
    categorical_cols = [
        col
        for col in features.columns
        if is_object_dtype(features[col]) or is_categorical_dtype(features[col])
    ]
    numeric_cols = [col for col in features.columns if col not in categorical_cols]
    numeric_df = (
        features[numeric_cols].fillna(0.0)
        if numeric_cols
        else pd.DataFrame(index=features.index)
    )
    total_rows = len(features)
    if numeric_cols:
        scaler = StandardScaler()
        numeric_train = scaler.fit_transform(numeric_df.iloc[:split_idx]).astype(
            np.float32
        )
        numeric_valid = scaler.transform(numeric_df.iloc[split_idx:]).astype(
            np.float32
        )
        scaler_mean = scaler.mean_.astype(np.float32)
        scaler_scale = scaler.scale_.astype(np.float32)
    else:
        numeric_train = np.zeros((split_idx, 0), dtype=np.float32)
        numeric_valid = np.zeros((total_rows - split_idx, 0), dtype=np.float32)
        scaler_mean = np.zeros((0,), dtype=np.float32)
        scaler_scale = np.ones((0,), dtype=np.float32)

    categorical_train: Dict[str, np.ndarray] = {}
    categorical_valid: Dict[str, np.ndarray] = {}
    cat_vocab_sizes: Dict[str, int] = {}
    for col in categorical_cols:
        series = features[col].fillna("__unknown__").astype(str)
        train_series = series.iloc[:split_idx].reset_index(drop=True)
        valid_series = series.iloc[split_idx:].reset_index(drop=True)
        vocab = {value: idx + 1 for idx, value in enumerate(train_series.unique())}
        cat_vocab_sizes[col] = len(vocab) + 1
        train_encoded = train_series.map(vocab).fillna(0).astype(np.int64).to_numpy()
        valid_encoded = valid_series.map(vocab).fillna(0).astype(np.int64).to_numpy()
        categorical_train[col] = train_encoded
        categorical_valid[col] = valid_encoded

    return StructuredFeatureMatrices(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        numeric_train=numeric_train,
        numeric_valid=numeric_valid,
        categorical_train=categorical_train,
        categorical_valid=categorical_valid,
        cat_vocab_sizes=cat_vocab_sizes,
        numeric_scaler_mean=scaler_mean,
        numeric_scaler_scale=scaler_scale,
    )


class HNMultimodalDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: AutoTokenizer,
        max_length: int,
        numeric_features: np.ndarray,
        categorical_features: Dict[str, np.ndarray],
        cat_feature_names: List[str],
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        numeric_array = np.asarray(numeric_features, dtype=np.float32)
        if numeric_array.ndim == 1:
            numeric_array = numeric_array.reshape(-1, 1)
        self.numeric_features = numeric_array
        self.numeric_dim = self.numeric_features.shape[1] if self.numeric_features.ndim == 2 else 0
        self.categorical_features = {
            name: np.asarray(values, dtype=np.int64)
            for name, values in categorical_features.items()
        }
        self.cat_feature_names = cat_feature_names
        if len(self.frame) != len(self.numeric_features):
            raise ValueError("Numeric feature matrix length mismatch with frame.")
        for name in self.cat_feature_names:
            if name not in self.categorical_features:
                raise KeyError(f"Missing categorical feature '{name}'.")
            if len(self.categorical_features[name]) != len(self.frame):
                raise ValueError(f"Categorical feature '{name}' length mismatch.")

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
        tokens["numeric_feats"] = self.numeric_features[idx]
        for name in self.cat_feature_names:
            tokens[f"cat_{name}"] = np.array(
                self.categorical_features[name][idx], dtype=np.int64
            )
        return tokens


class MultimodalDataCollator:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        pad_to_multiple_of: int | None,
        cat_feature_names: List[str],
    ) -> None:
        self.text_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
        )
        self.cat_feature_names = cat_feature_names

    def __call__(self, features: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        extra_keys = {"numeric_feats", *[f"cat_{name}" for name in self.cat_feature_names]}
        text_features = []
        for feat in features:
            text_features.append({k: v for k, v in feat.items() if k not in extra_keys})
        batch = self.text_collator(text_features)
        numeric_stack = np.stack(
            [np.asarray(feat["numeric_feats"], dtype=np.float32) for feat in features],
            axis=0,
        )
        batch["numeric_feats"] = torch.tensor(numeric_stack, dtype=torch.float32)
        for name in self.cat_feature_names:
            key = f"cat_{name}"
            cat_values: List[int] = []
            for feat in features:
                arr = np.asarray(feat[key])
                cat_values.append(int(arr.reshape(-1)[0]))
            batch[key] = torch.tensor(cat_values, dtype=torch.long)
        return batch


class HNMultiModalClassifier(nn.Module):
    def __init__(
        self,
        model_name: str,
        numeric_dim: int,
        categorical_vocab_sizes: Dict[str, int],
        tabular_hidden_dim: int = 128,
        max_cat_embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)
        self.numeric_dim = numeric_dim
        self.cat_feature_names = list(categorical_vocab_sizes.keys())
        dropout = getattr(self.config, "hidden_dropout_prob", 0.1)
        self.text_dropout = nn.Dropout(dropout)
        self.cat_embeddings = nn.ModuleDict()
        for name, vocab_size in categorical_vocab_sizes.items():
            emb_dim = min(
                max_cat_embedding_dim,
                max(4, int(math.ceil(math.log2(vocab_size + 1)))),
            )
            self.cat_embeddings[name] = nn.Embedding(
                vocab_size,
                emb_dim,
                padding_idx=0,
            )
        tabular_input_dim = numeric_dim + sum(
            embedding.embedding_dim for embedding in self.cat_embeddings.values()
        )
        if tabular_input_dim > 0 and tabular_hidden_dim > 0:
            self.tabular_mlp = nn.Sequential(
                nn.LayerNorm(tabular_input_dim),
                nn.Linear(tabular_input_dim, tabular_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            tabular_dim = tabular_hidden_dim
        elif tabular_input_dim > 0:
            self.tabular_mlp = None
            tabular_dim = tabular_input_dim
        else:
            self.tabular_mlp = None
            tabular_dim = 0
        fusion_dim = self.config.hidden_size + tabular_dim
        head_hidden = max(64, fusion_dim // 2) if fusion_dim >= 64 else fusion_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, numeric_feats=None, **kwargs):
        cat_keys = {f"cat_{name}" for name in self.cat_feature_names}
        excluded = cat_keys.union({"numeric_feats", "labels"})
        text_inputs = {
            key: value
            for key, value in kwargs.items()
            if key not in excluded
        }
        outputs = self.transformer(**text_inputs, return_dict=True)
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            text_repr = pooler_output
        else:
            text_repr = outputs.last_hidden_state[:, 0]
        text_repr = self.text_dropout(text_repr)
        tabular_chunks = []
        if numeric_feats is not None and self.numeric_dim > 0:
            tabular_chunks.append(numeric_feats)
        for name in self.cat_feature_names:
            cat_key = f"cat_{name}"
            if cat_key in kwargs:
                tabular_chunks.append(self.cat_embeddings[name](kwargs[cat_key]))
        if tabular_chunks:
            tabular_input = torch.cat(tabular_chunks, dim=-1)
            tabular_repr = (
                self.tabular_mlp(tabular_input)
                if self.tabular_mlp is not None
                else tabular_input
            )
            fused = torch.cat([text_repr, tabular_repr], dim=-1)
        else:
            fused = text_repr
        logits = self.classifier(fused)
        return SequenceClassifierOutput(logits=logits)


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
        if hasattr(eval_pred, "predictions"):
            logits = eval_pred.predictions
            labels = eval_pred.label_ids
        else:
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
    cleaned = cleaned.sort_values("time").reset_index(drop=True)
    print(
        f"Loaded {len(raw):,} rows → {len(cleaned):,} after URL dedup "
        f"(removed {data_stats['deduplicated']:,})."
    )
    baseline = cleaned["label"].mean()
    print(f"Class balance (viral proportion): {baseline:.2%}")

    train_df, valid_df, split_idx = chronological_split(cleaned, args.train_fraction)
    print(
        f"Chronological split sizes — train: {len(train_df):,}, "
        f"valid: {len(valid_df):,}"
    )
    structured_spec = prepare_structured_features(
        cleaned,
        split_idx=split_idx,
        viral_threshold=args.viral_threshold,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    cat_train_features = {
        name: structured_spec.categorical_train[name]
        for name in structured_spec.categorical_cols
    }
    cat_valid_features = {
        name: structured_spec.categorical_valid[name]
        for name in structured_spec.categorical_cols
    }
    train_dataset = HNMultimodalDataset(
        frame=train_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        numeric_features=structured_spec.numeric_train,
        categorical_features=cat_train_features,
        cat_feature_names=structured_spec.categorical_cols,
    )
    valid_dataset = HNMultimodalDataset(
        frame=valid_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        numeric_features=structured_spec.numeric_valid,
        categorical_features=cat_valid_features,
        cat_feature_names=structured_spec.categorical_cols,
    )

    collator = MultimodalDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None,
        cat_feature_names=structured_spec.categorical_cols,
    )
    pos_weight = compute_pos_weight(train_df["label"].values)
    print(f"Positive class weight: {pos_weight:.2f}")
    model = HNMultiModalClassifier(
        model_name=args.model_name,
        numeric_dim=structured_spec.numeric_train.shape[1],
        categorical_vocab_sizes={
            name: structured_spec.cat_vocab_sizes[name]
            for name in structured_spec.categorical_cols
        },
        tabular_hidden_dim=args.tabular_hidden_dim,
        max_cat_embedding_dim=args.max_cat_embed_dim,
    )

    training_args = TrainingArguments(
        output_dir=str(args.reports_dir / "checkpoints"),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        do_eval=True,
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
        remove_unused_columns=False,
        label_names=["labels"],
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
            "tabular_hidden_dim": args.tabular_hidden_dim,
            "max_cat_embed_dim": args.max_cat_embed_dim,
        },
        "structured_features": {
            "numeric_columns": structured_spec.numeric_cols,
            "categorical_columns": structured_spec.categorical_cols,
            "categorical_vocab_sizes": {
                key: int(value)
                for key, value in structured_spec.cat_vocab_sizes.items()
            },
            "numeric_scaler_stats": {
                col: {
                    "mean": float(mean),
                    "scale": float(scale),
                }
                for col, mean, scale in zip(
                    structured_spec.numeric_cols,
                    structured_spec.numeric_scaler_mean.tolist(),
                    structured_spec.numeric_scaler_scale.tolist(),
                )
            },
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

