import json
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from catboost import CatBoostClassifier

from cog import BasePredictor, Input 
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

        viral_threshold = int(os.environ.get("VIRAL_SCORE_THRESHOLD", "250"))
        title_embedding_model = os.environ.get(
            "TITLE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        cache_path_value = os.environ.get("TITLE_EMBEDDING_CACHE")
        cache_path = Path(cache_path_value) if cache_path_value else None

        stats_path = Path(os.environ.get("FEATURE_STATS_PATH", "reports/feature_stats.json"))
        if stats_path.exists():
            with open(stats_path) as f:
                stats_store = json.load(f)
        else:
            raise FileNotFoundError(
                f"Feature stats not found at {stats_path}. "
                "Inference cannot proceed without historical features. "
                "Run `src/feature_stats.py` to generate them."
            )

        self.feature_engineer = FeatureEngineer(
            viral_threshold=viral_threshold,
            title_embedding_model=title_embedding_model,
            title_embedding_cache_path=cache_path,
            stats_store=stats_store,
        )
        self.categorical_features = ("by", "domain")

    def predict(
        self,
        title: str = Input(description="Post title text.", default="AI is going to take over the world"),
        url: str = Input(
            default="https://openai.com",
            description="Post URL (empty string for self posts).",
        ),
        by: str = Input(
            default="zama",
            description="Author username.",
        ),
        time: int = Input(
            default=1763807356,
            ge=0,
            description="Unix timestamp (seconds) when the post was created.",
        ),
    ) -> Dict[str, Any]:
        """Score a single post constructed from the provided scalar inputs."""
        raw = self._build_single_row(title=title, url=url, by=by, time=time)
        bundle = self.feature_engineer.transform(raw)
        features = bundle.features
        if features.empty:
            return {"probability": None}

        probabilities = self.model.predict_proba(
            features,
        )[:, 1]

        probability = float(probabilities[0])
        return {
            "probability": probability,
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

