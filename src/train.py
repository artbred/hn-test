from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

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
        help="Minimum recall to satisfy when using precision_at_recall strategy.",
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=0,
        help="Number of Optuna trials to run for CatBoost hyperparameters. 0 disables tuning.",
    )
    parser.add_argument(
        "--optuna-metric",
        choices=("roc_auc", "pr_auc", "f1"),
        default="pr_auc",
        help="Validation metric Optuna should maximize.",
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


def build_catboost_classifier(
    class_weights: List[float], **overrides
) -> CatBoostClassifier:
    params = {
        "depth": 6,
        "learning_rate": 0.015,
        "iterations": 600,
        "bagging_temperature": 4.31,
        "random_strength": 1.23,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": 42,
        "early_stopping_rounds": 50,
        "l2_leaf_reg": 6.45,
        "border_count": 64,
        "class_weights": list(class_weights),
        "use_best_model": True,
        "verbose": 100,
        "allow_writing_files": False,
    }
    params.update(overrides)
    return CatBoostClassifier(**params)


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
    prob_columns: Dict[str, np.ndarray],
) -> None:
    frame = pd.DataFrame({"id": ids})
    for name, values in prob_columns.items():
        frame[name] = values
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


def tune_threshold(
    y_true: pd.Series,
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
    thresholds = thresholds

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


def select_dense_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return columns safe for dense models (numeric only)."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns available for dense models.")
    return numeric_cols


def dense_matrix(df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    return df[columns].astype(np.float32).values


def summarize_model_metrics(
    y_true: pd.Series,
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


def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cat_feature_idx: List[int],
    class_weights: List[float],
    metric_name: str,
    n_trials: int,
    target_recall: float,
) -> Dict[str, float]:
    if n_trials <= 0:
        return {}

    try:
        optuna = importlib.import_module("optuna")
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "Optuna is required when --optuna-trials > 0. Install it via `pip install optuna`."
        ) from exc

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial) -> float:
        params = {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
            "iterations": trial.suggest_int("iterations", 200, 1000),
            "border_count": trial.suggest_int("border_count", 64, 254),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 2.0, log=True),
        }
        model = build_catboost_classifier(
            class_weights,
            **params,
            verbose=0,
        )
        model.fit(
            X_train,
            y_train,
            cat_features=cat_feature_idx,
            eval_set=(X_valid, y_valid),
            verbose=False,
        )
        probs = model.predict_proba(X_valid)[:, 1]
        roc = roc_auc_score(y_valid, probs)
        pr = average_precision_score(y_valid, probs)

        if metric_name == "roc_auc":
            value = roc
        elif metric_name == "pr_auc":
            value = pr
        else:
            threshold, _ = tune_threshold(
                y_valid, probs, strategy="f1", target_recall=target_recall
            )
            preds = (probs >= threshold).astype(int)
            value = f1_score(y_valid, preds)

        trial.set_user_attr("roc_auc", float(roc))
        trial.set_user_attr("pr_auc", float(pr))
        return float(value)

    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(
        f"Optuna best {metric_name}: {study.best_value:.4f} "
        f"with params: {study.best_params}"
    )
    return study.best_params


def main() -> None:
    args = parse_args()
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    viral_threshold = 250

    raw = load_raw_posts(args.data_path, n_rows=args.n_rows)
    print(f"Loaded {len(raw):,} posts with columns: {list(raw.columns)}")
    viral_rate = (raw["score"] > viral_threshold).mean()
    print(f"Baseline viral rate (>={viral_threshold} score): {viral_rate:.2%}")

    engineer = FeatureEngineer(
        viral_threshold=viral_threshold,
        title_embedding_cache_path=args.title_embedding_cache,
    )
    bundle = engineer.transform(raw)

    splits = chronological_split(bundle, args.train_fraction)
    class_weights = compute_class_weights(splits["y_train"])
    cat_feature_idx = [
        splits["X_train"].columns.get_loc(col)
        for col in ["type", "domain", "by"]
        if col in splits["X_train"].columns
    ]
    dense_feature_cols = select_dense_feature_columns(splits["X_train"])

    optuna_params: Dict[str, float | int] = {}
    if args.optuna_trials > 0:
        print(
            f"Running Optuna for {args.optuna_trials} trials optimizing {args.optuna_metric}..."
        )
        optuna_params = run_optuna_search(
            splits["X_train"],
            splits["y_train"],
            splits["X_valid"],
            splits["y_valid"],
            cat_feature_idx=cat_feature_idx,
            class_weights=class_weights["catboost"],
            metric_name=args.optuna_metric,
            n_trials=args.optuna_trials,
            target_recall=args.target_recall,
        )

    cat_model = build_catboost_classifier(
        class_weights["catboost"],
        **optuna_params,
    )

    print("Training CatBoost...")
    cat_model.fit(
        splits["X_train"],
        splits["y_train"],
        cat_features=cat_feature_idx,
        eval_set=(splits["X_valid"], splits["y_valid"]),
    )
    cat_valid_probs = cat_model.predict_proba(splits["X_valid"])[:, 1]


    metrics = {}
    y_valid = splits["y_valid"]
    for name, probs in [
        ("catboost", cat_valid_probs),
    ]:
        _, model_metrics = summarize_model_metrics(
            y_valid,
            probs,
            strategy=args.threshold_strategy,
            target_recall=args.target_recall,
        )
        metrics[name] = model_metrics

    if args.optuna_trials > 0 and optuna_params:
        metrics["catboost"]["optuna_trials"] = int(args.optuna_trials)
        metrics["catboost"]["optuna_metric"] = args.optuna_metric
        metrics["catboost"]["optuna_best_params"] = {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else int(value)
                if isinstance(value, (int, np.integer))
                else value
            )
            for key, value in optuna_params.items()
        }

    metrics_path = args.reports_dir / "metrics.json"
    preds_path = args.reports_dir / "validation_predictions.csv"
    cat_imp_path = args.reports_dir / "catboost_feature_importance.csv"

    save_metrics(metrics_path, metrics)
    save_predictions(
        preds_path,
        splits["ids_valid"],
        {
            "catboost_prob": cat_valid_probs
        },
    )
    save_feature_importance(cat_model, cat_imp_path, cat_features=list(splits["X_train"].columns))

    print(f"Artifacts saved under {args.reports_dir}")


if __name__ == "__main__":
    main()


