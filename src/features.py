from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    ids: pd.Series


def load_raw_posts(csv_path: Path, n_rows: int | None = None) -> pd.DataFrame:
    """Load the Hacker News posts CSV."""
    df = pd.read_csv(csv_path, nrows=n_rows)
    expected = {"id", "title", "url", "type", "by", "time", "score"}
    missing = expected.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df


class FeatureEngineer:
    """Build leakage-safe features for HN virality modeling."""

    def __init__(self, viral_threshold: int = 500) -> None:
        self.viral_threshold = viral_threshold
        self._extractor = tldextract.TLDExtract(include_psl_private_domains=True)

    def transform(self, df: pd.DataFrame) -> DatasetBundle:
        work = df.copy()
        work["is_viral"] = (work["score"] > self.viral_threshold).astype(np.int8)
        work["title"] = work["title"].fillna("")
        work["url"] = work["url"].fillna("")
        work["text"] = work["text"].fillna("")

        work = self._add_temporal_features(work)
        work = self._add_url_features(work)

        work = self._add_historical_features(work, group_col="domain", prefix="domain")
        work = self._add_historical_features(work, group_col="by", prefix="user")

        feature_cols = self._select_feature_columns(work)
        features = work[feature_cols].copy()
        self._finalize_feature_dtypes(features)

        target = work["is_viral"].astype(np.int8)
        ids = work["id"]
        return DatasetBundle(features=features, target=target, ids=ids)

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        posted_at = pd.to_datetime(df["time"], unit="s", utc=True)
        df["posted_hour"] = posted_at.dt.hour.astype(np.int8)
        df["posted_dayofweek"] = posted_at.dt.dayofweek.astype(np.int8)
        return df

    def _add_url_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["domain"] = df["url"].apply(self._extract_domain).fillna("unknown")
        df["is_self_post"] = (df["url"] == "").astype(np.int8)
        df["url_length"] = df["url"].str.len().astype(np.int32)
        df["has_query"] = df["url"].str.contains(r"\?").astype(np.int8)
        df["path_depth"] = df["url"].apply(self._path_depth).astype(np.int8)
        df["is_root_domain"] = (df["path_depth"] == 0).astype(np.int8)
        df["domain_token_len"] = df["domain"].str.split(".").str.len().fillna(0).astype(
            np.int8
        )
        return df


    def _add_historical_features(
        self, df: pd.DataFrame, group_col: str, prefix: str
    ) -> pd.DataFrame:
        working = df[[group_col, "time", "is_viral", "score"]].copy()
        working = working.sort_values("time")
        grouped = working.groupby(group_col, sort=False, dropna=False)

        prior_posts = grouped.cumcount()
        prior_viral = grouped["is_viral"].cumsum() - working["is_viral"]
        prior_scores = grouped["score"].cumsum() - working["score"]
        prior_hours = grouped["time"].diff() / 3600.0

        stats = pd.DataFrame(
            {
                f"{prefix}_prior_posts": prior_posts,
                f"{prefix}_prior_viral_count": prior_viral,
                f"{prefix}_prior_mean_score": prior_scores
                / prior_posts.replace(0, np.nan),
                f"{prefix}_prior_viral_rate": prior_viral
                / prior_posts.replace(0, np.nan),
                f"{prefix}_hours_since_last": prior_hours,
            },
            index=working.index,
        )

        stats[f"{prefix}_prior_posts"] = stats[f"{prefix}_prior_posts"].astype(
            np.float32
        )
        stats[f"{prefix}_prior_posts_log1p"] = np.log1p(
            stats[f"{prefix}_prior_posts"].clip(lower=0)
        ).astype(np.float32)
        stats[f"{prefix}_prior_viral_count"] = stats[
            f"{prefix}_prior_viral_count"
        ].astype(np.float32)
        stats[f"{prefix}_prior_mean_score"] = stats[
            f"{prefix}_prior_mean_score"
        ].clip(lower=0).astype(np.float32)
        stats[f"{prefix}_prior_viral_rate"] = stats[
            f"{prefix}_prior_viral_rate"
        ].clip(lower=0, upper=1).astype(np.float32)
        stats[f"{prefix}_hours_since_last"] = stats[
            f"{prefix}_hours_since_last"
        ].astype(np.float32)

        stats = stats.reindex(df.index)
        overall_viral_rate = df["is_viral"].mean()
        overall_score = df["score"].mean()
        median_hours = stats[f"{prefix}_hours_since_last"].median()
        if np.isnan(median_hours):
            median_hours = 24.0
        stats[f"{prefix}_prior_mean_score"] = stats[
            f"{prefix}_prior_mean_score"
        ].fillna(overall_score)
        stats[f"{prefix}_prior_viral_rate"] = stats[
            f"{prefix}_prior_viral_rate"
        ].fillna(overall_viral_rate)
        stats[f"{prefix}_hours_since_last"] = stats[f"{prefix}_hours_since_last"].fillna(
            median_hours
        )

        for col in stats.columns:
            df[col] = stats[col]
        return df

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        base_cols = [
            "url_length",
            "has_query",
            "path_depth",
            "is_root_domain",
            "is_self_post",
            "domain_token_len",
            "domain_prior_posts",
            "domain_prior_posts_log1p",
            "domain_prior_viral_count",
            "domain_prior_mean_score",
            "domain_prior_viral_rate",
            "domain_hours_since_last",
            "user_prior_posts",
            "user_prior_posts_log1p",
            "user_prior_viral_count",
            "user_prior_mean_score",
            "user_prior_viral_rate",
            "user_hours_since_last",
        ]
        categorical = ["domain", "by"]
        return base_cols + categorical

    def _finalize_feature_dtypes(self, features: pd.DataFrame) -> None:
        for col in features.select_dtypes(include=["float64"]).columns:
            features[col] = features[col].astype(np.float32)
        for col in features.select_dtypes(include=["int64"]).columns:
            features[col] = features[col].astype(np.int32)

    def _extract_domain(self, url: str) -> str:
        if not isinstance(url, str) or not url.strip():
            return "self-post"
        parsed = self._extractor(url)
        if parsed.domain and parsed.suffix:
            return f"{parsed.domain}.{parsed.suffix}".lower()
        netloc = urlparse(url).netloc.lower()
        return netloc or "unknown"

    @staticmethod
    def _path_depth(url: str) -> int:
        if not isinstance(url, str) or not url:
            return 0
        path = urlparse(url).path
        parts = [segment for segment in path.split("/") if segment]
        return len(parts)


