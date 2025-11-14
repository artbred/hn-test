from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from catboost import CatBoostClassifier

try:  # pragma: no cover - allows local dev without Cog installed
    from cog import BasePredictor, Input  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    class BasePredictor:  # type: ignore[misc]
        pass

    class Input:  # type: ignore[misc]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

from src.features import FeatureEngineer


class Predictor(BasePredictor):
    """Cog predictor that scores Hacker News posts with the CatBoost model."""

    def setup(self) -> None:
        """Load the CatBoost model and supporting artifacts once per container."""
        model_path = Path(
            os.environ.get("CATBOOST_MODEL_PATH", "reports/catboost_model.cbm")
        )
        if not model_path.exists():
            raise FileNotFoundError(
                f"CatBoost model not found at {model_path}. "
                "Export the trained model with `CatBoostClassifier.save_model`."
            )

        self.model = CatBoostClassifier()
        self.model.load_model(str(model_path))

        metrics_path = Path(
            os.environ.get("CATBOOST_METRICS_PATH", "reports/metrics.json")
        )
        self.default_threshold = 0.5
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
                self.default_threshold = float(
                    metrics.get("catboost", {}).get(
                        "decision_threshold", self.default_threshold
                    )
                )
            except (json.JSONDecodeError, ValueError):
                # Fallback to the default threshold when the metrics file is corrupt.
                pass

        viral_threshold = int(os.environ.get("VIRAL_SCORE_THRESHOLD", "250"))
        title_embedding_model = os.environ.get(
            "TITLE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        cache_path_value = os.environ.get("TITLE_EMBEDDING_CACHE")
        cache_path = Path(cache_path_value) if cache_path_value else None

        self.feature_engineer = FeatureEngineer(
            viral_threshold=viral_threshold,
            title_embedding_model=title_embedding_model,
            title_embedding_cache_path=cache_path,
        )
        self.categorical_features = ("by", "domain")

    def predict(
        self,
        title: str = Input(description="Post title text."),
        url: str = Input(
            default="",
            description="Post URL (empty string for self posts).",
        ),
        by: str = Input(
            default="unknown",
            description="Author username.",
        ),
        time: int = Input(
            ge=0,
            description="Unix timestamp (seconds) when the post was created.",
        ),
    ) -> Dict[str, Any]:
        """Score a single post constructed from the provided scalar inputs."""
        raw = self._build_single_row(title=title, url=url, by=by, time=time)
        bundle = self.feature_engineer.transform(raw)
        features = bundle.features
        if features.empty:
            return {"threshold": self.default_threshold, "prediction": None}

        cat_feature_idx = [
            features.columns.get_loc(col)
            for col in self.categorical_features
            if col in features.columns
        ]
        probabilities = self.model.predict_proba(
            features,
            cat_features=cat_feature_idx if cat_feature_idx else None,
        )[:, 1]

        probability = float(probabilities[0])
        is_viral = bool(probability >= self.default_threshold)
        return {
            "threshold": self.default_threshold,
            "prediction": {
                "probability": probability,
                "is_viral": is_viral,
            },
        }

    def _build_single_row(self, title: str, url: str, by: str, time: int) -> pd.DataFrame:
        try:
            timestamp = int(time)
        except (TypeError, ValueError) as exc:
            raise ValueError("`time` must be an integer Unix timestamp.") from exc

        row = {
            "id": 0,
            "title": title or "",
            "url": url or "",
            "type": "story",
            "by": by or "unknown",
            "time": timestamp,
            "score": 0,
            "text": "",
        }
        return pd.DataFrame([row])

