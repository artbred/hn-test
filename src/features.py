from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from embeddings import compute_sentence_embeddings


@dataclass
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    ids: pd.Series
    timestamps: pd.Series
    embedding_features: pd.DataFrame | None = None


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

    def __init__(
        self,
        viral_threshold: int = 500,
        title_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        title_embedding_cache_path: str | Path | None = None,
        title_embedding_batch_size: int = 256,
        title_embedding_normalize: bool = True,
        title_embedding_dim: int | None = 32,
        title_embedding_scale: bool = True,
    ) -> None:
        self.viral_threshold = viral_threshold
        self._extractor = tldextract.TLDExtract(include_psl_private_domains=True)
        self.title_embedding_model = title_embedding_model
        self.title_embedding_cache_path = (
            Path(title_embedding_cache_path)
            if title_embedding_cache_path is not None
            else None
        )
        self.title_embedding_batch_size = title_embedding_batch_size
        self.title_embedding_normalize = title_embedding_normalize
        self.title_embedding_dim = title_embedding_dim
        self.title_embedding_scale = title_embedding_scale
        self._title_embedding_cols: List[str] = []

    def transform(self, df: pd.DataFrame) -> DatasetBundle:
        work = df.copy()
        work["is_viral"] = (work["score"] > self.viral_threshold).astype(np.int8)
        work["is_top_1pct"] = self._mark_top_fraction(
            scores=work["score"], fraction=0.01
        )
        work["title"] = work["title"].fillna("")
        work["url"] = work["url"].fillna("")
        work["text"] = work["text"].fillna("")

        work = self._add_temporal_features(work)
        work = self._add_url_features(work)
        work = work.copy()
        work, self._title_embedding_cols = self._add_title_embeddings(work)

        work = self._add_historical_features(work, group_col="domain", prefix="domain")
        work = self._add_historical_features(
            work,
            group_col="by",
            prefix="user",
            top_flag_col="is_top_1pct",
            top_flag_output="user_prior_top1pct_count",
        )
        work = self._add_historical_features(
            work,
            group_col=["by", "domain"],
            prefix="user_domain",
        )

        work["user_vs_domain_viral_rate_gap"] = (
            work["user_prior_viral_rate"] - work["domain_prior_viral_rate"]
        ).astype(np.float32)
        work["user_vs_domain_mean_score_gap"] = (
            work["user_prior_mean_score"] - work["domain_prior_mean_score"]
        ).astype(np.float32)

        embedding_features = (
            work[self._title_embedding_cols].copy()
            if self._title_embedding_cols
            else pd.DataFrame(index=work.index)
        )
        feature_cols = self._select_feature_columns(work)
        features = work[feature_cols].copy()
        self._finalize_feature_dtypes(features)

        target = work["is_viral"].astype(np.int8)
        ids = work["id"]
        timestamps = work["time"].copy()
        return DatasetBundle(
            features=features,
            target=target,
            ids=ids,
            timestamps=timestamps,
            embedding_features=embedding_features,
        )

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
        self,
        df: pd.DataFrame,
        group_col: str | Sequence[str],
        prefix: str,
        top_flag_col: str | None = None,
        top_flag_output: str | None = None,
    ) -> pd.DataFrame:
        group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
        cols = [*group_cols, "time", "is_viral", "score"]
        if top_flag_col and top_flag_col not in cols:
            cols.append(top_flag_col)
        working = df[cols].copy()
        working = working.sort_values("time")
        grouped = working.groupby(group_cols, sort=False, dropna=False)
        if len(group_cols) == 1:
            groupby_keys = working[group_cols[0]]
        else:
            groupby_keys = [working[col] for col in group_cols]

        prior_posts = grouped.cumcount()
        prior_viral = grouped["is_viral"].cumsum() - working["is_viral"]
        prior_scores = grouped["score"].cumsum() - working["score"]
        prior_hours = grouped["time"].diff() / 3600.0
        prior_mean_score = prior_scores / prior_posts.replace(0, np.nan)
        shifted_scores = grouped["score"].shift(1)
        prior_max_score = shifted_scores.groupby(
            groupby_keys, sort=False, dropna=False
        ).cummax()

        data = {
            f"{prefix}_prior_posts": prior_posts,
            f"{prefix}_prior_viral_count": prior_viral,
            f"{prefix}_prior_mean_score": prior_mean_score,
            f"{prefix}_prior_viral_rate": prior_viral / prior_posts.replace(0, np.nan),
            f"{prefix}_hours_since_last": prior_hours,
            f"{prefix}_prior_max_score": prior_max_score,
        }
        top_flag_feature = top_flag_output or (
            f"{prefix}_prior_top_flag_count" if top_flag_col else None
        )
        if top_flag_col and top_flag_feature:
            prior_top_flag = grouped[top_flag_col].cumsum() - working[top_flag_col]
            data[top_flag_feature] = prior_top_flag

        stats = pd.DataFrame(data, index=working.index)

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
        stats[f"{prefix}_prior_max_score"] = stats[
            f"{prefix}_prior_max_score"
        ].astype(np.float32)
        if top_flag_feature:
            stats[top_flag_feature] = stats[top_flag_feature].astype(np.float32)

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
        stats[f"{prefix}_prior_max_score"] = stats[f"{prefix}_prior_max_score"].fillna(
            overall_score
        )
        if top_flag_feature:
            stats[top_flag_feature] = stats[top_flag_feature].fillna(0.0)

        if prefix in {"user", "user_domain"}:
            stats[f"{prefix}_avg_score"] = stats[f"{prefix}_prior_mean_score"].astype(
                np.float32
            )
            stats[f"{prefix}_mean_score"] = stats[f"{prefix}_prior_mean_score"].astype(
                np.float32
            )
            stats[f"{prefix}_max_score"] = stats[f"{prefix}_prior_max_score"].astype(
                np.float32
            )

        return df.join(stats, how="left")
        
    def _mark_top_fraction(self, scores: pd.Series, fraction: float) -> pd.Series:
        if scores.empty:
            return pd.Series(np.zeros(0, dtype=np.int8), index=scores.index)
        top_k = max(int(np.ceil(len(scores) * fraction)), 1)
        top_scores = scores.nlargest(top_k)
        threshold = top_scores.min()
        mask = (scores >= threshold).astype(np.int8)
        return mask


    def _add_title_embeddings(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        if not self.title_embedding_model:
            return df, []

        embeddings = compute_sentence_embeddings(
            texts=df["title"].tolist(),
            model_name=self.title_embedding_model,
            batch_size=self.title_embedding_batch_size,
            cache_path=self.title_embedding_cache_path,
            normalize=self.title_embedding_normalize,
        )
        if embeddings.shape[0] != len(df):
            raise ValueError(
                "Embedding row count does not match dataframe length: "
                f"{embeddings.shape[0]} vs {len(df)}"
            )

        matrix = embeddings.astype(np.float32)

        if self.title_embedding_scale:
            scaler = StandardScaler()
            matrix = scaler.fit_transform(matrix).astype(np.float32)

        reduced_dim = self.title_embedding_dim
        if reduced_dim is not None and reduced_dim < matrix.shape[1]:
            reducer = TruncatedSVD(
                n_components=reduced_dim,
                random_state=42,
            )
            matrix = reducer.fit_transform(matrix).astype(np.float32)

        col_names = [f"title_emb_{i:03d}" for i in range(matrix.shape[1])]
        embedding_df = pd.DataFrame(matrix, columns=col_names, index=df.index)
        df = df.join(embedding_df)
        return df, col_names

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        base_cols = [
            "posted_hour",
            "posted_dayofweek",
            "user_avg_score",
            "user_mean_score",
            "user_max_score",
            "user_domain_avg_score",
            "user_domain_mean_score",
            "user_domain_max_score",
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

    def get_title_embedding_columns(self) -> List[str]:
        return list(self._title_embedding_cols)