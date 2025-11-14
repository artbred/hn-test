from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from features import DatasetBundle, FeatureEngineer, load_raw_posts


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train CatBoost + XGBoost ensemble to predict viral HN posts."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=repo_root / "data" / "hn_posts.csv",
        help="Absolute path to the hn_posts.csv file.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="If provided, limit the loaded dataset to the first N rows.",
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
        "--title-embedding-cache",
        type=Path,
        default=repo_root / "data" / "embeddings" / "title_minilm.npy",
        help="Path to cache sentence-transformer embeddings (.npy).",
    )
    return parser.parse_args()


def chronological_split(
    bundle: DatasetBundle, train_fraction: float
) -> Dict[str, pd.DataFrame | pd.Series]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    order = np.argsort(bundle.timestamps.values)
    X_sorted = bundle.features.iloc[order].reset_index(drop=True)
    y_sorted = bundle.target.iloc[order].reset_index(drop=True)
    ids_sorted = bundle.ids.iloc[order].reset_index(drop=True)

    split_idx = int(len(X_sorted) * train_fraction)
    return {
        "X_train": X_sorted.iloc[:split_idx],
        "y_train": y_sorted.iloc[:split_idx],
        "X_valid": X_sorted.iloc[split_idx:],
        "y_valid": y_sorted.iloc[split_idx:],
        "ids_valid": ids_sorted.iloc[split_idx:],
    }


def build_catboost_classifier(class_weights: List[float]) -> CatBoostClassifier:
    return CatBoostClassifier(
        depth=8,
        learning_rate=0.08,
        iterations=600,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        early_stopping_rounds=50,
        l2_leaf_reg=5.0,
        border_count=254,
        class_weights=list(class_weights),
        use_best_model=True,
        verbose=100,
        allow_writing_files=False,
    )


def compute_class_weights(y: pd.Series) -> Dict[str, float]:
    pos_rate = y.mean()
    pos_weight = float((1 - pos_rate) / pos_rate)
    return {"catboost": [1.0, pos_weight]}


def evaluate_metrics(
    y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    preds = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, preds)),
        "accuracy": float(accuracy_score(y_true, preds)),
    }


def save_metrics(path: Path, metrics: Dict[str, Dict[str, float]]) -> None:
    path.write_text(json.dumps(metrics, indent=2))


def save_predictions(
    path: Path,
    ids: pd.Series,
    cat_probs: np.ndarray,
) -> None:
    frame = pd.DataFrame(
        {
            "id": ids,
            "catboost_prob": cat_probs,
        }
    )
    frame.to_csv(path, index=False)


def save_feature_importance(
    cat_model: CatBoostClassifier,
    cat_path: Path,
    cat_features: List[str],
) -> None:
    cat_importance = pd.DataFrame(
        {
            "feature": cat_features,
            "importance": cat_model.get_feature_importance(type="FeatureImportance"),
        }
    ).sort_values("importance", ascending=False)
    cat_importance.to_csv(cat_path, index=False)

def write_summary(
    path: Path,
    dataset_shape: tuple[int, int],
    viral_rate: float,
    metrics: Dict[str, Dict[str, float]],
) -> None:
    lines = [
        "# HN Virality Modeling Summary",
        "",
        f"- Samples: {dataset_shape[0]:,} rows, {dataset_shape[1]} engineered features",
        f"- Viral threshold: score > 500, baseline rate {viral_rate:.2%}",
    ]
    lines.append("")
    lines.append("## Validation Metrics")
    for model_name, values in metrics.items():
        lines.append(f"- {model_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in values.items()))
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw_posts(args.data_path, n_rows=args.n_rows)
    print(f"Loaded {len(raw):,} posts with columns: {list(raw.columns)}")
    viral_rate = (raw["score"] > 500).mean()
    print(f"Baseline viral rate (>500 score): {viral_rate:.2%}")

    engineer = FeatureEngineer(
        viral_threshold=500,
        title_embedding_cache_path=args.title_embedding_cache,
    )
    bundle = engineer.transform(raw)

    splits = chronological_split(bundle, args.train_fraction)
    class_weights = compute_class_weights(splits["y_train"])
    cat_model = build_catboost_classifier(class_weights["catboost"])
    cat_cols = [
        col for col in ["type", "domain", "by"] if col in splits["X_train"].columns
    ]

    print("Training CatBoost...")
    cat_model.fit(
        splits["X_train"],
        splits["y_train"],
        cat_features=[splits["X_train"].columns.get_loc(col) for col in cat_cols],
        eval_set=(splits["X_valid"], splits["y_valid"]),
    )
    cat_valid_probs = cat_model.predict_proba(splits["X_valid"])[:, 1]

    metrics = {
        "catboost": evaluate_metrics(splits["y_valid"], cat_valid_probs),
    }

    metrics_path = args.reports_dir / "metrics.json"
    preds_path = args.reports_dir / "validation_predictions.csv"
    summary_path = args.reports_dir / "summary.md"
    cat_imp_path = args.reports_dir / "catboost_feature_importance.csv"

    save_metrics(metrics_path, metrics)
    save_predictions(preds_path, splits["ids_valid"], cat_valid_probs)
    save_feature_importance(cat_model, cat_imp_path, cat_features=list(splits["X_train"].columns))
    write_summary(
        summary_path,
        dataset_shape=bundle.features.shape,
        viral_rate=viral_rate,
        metrics=metrics,
    )

    print(f"Artifacts saved under {args.reports_dir}")


if __name__ == "__main__":
    main()


