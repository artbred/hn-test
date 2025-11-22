
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Sequence, List
import pandas as pd
import numpy as np

def calculate_stats(df: pd.DataFrame, group_cols: str | Sequence[str], prefix: str, top_flag_col: str | None = None) -> Dict[str, Any]:
    # We need: count, sum(is_viral), sum(score), max(score), max(time)
    # And if top_flag_col: sum(top_flag_col)
    
    aggs = {
        "post_count": ("id", "count"),
        "viral_count": ("is_viral", "sum"),
        "total_score": ("score", "sum"),
        "max_score": ("score", "max"),
        "last_time": ("time", "max"),
    }
    
    if top_flag_col:
        aggs["top_flag_count"] = (top_flag_col, "sum")
        
    grouped = df.groupby(group_cols).agg(**aggs)
    
    # Convert to dict keyed by group
    # If group_cols is list, index is MultiIndex. We'll convert to string key or nested dict.
    # For JSON serialization, string keys "val1" or "val1|val2" are easiest.
    
    stats = {}
    for idx, row in grouped.iterrows():
        key = str(idx) if not isinstance(idx, tuple) else "|".join(map(str, idx))
        
        def safe_int(val):
            try:
                return int(val)
            except (ValueError, TypeError):
                return 0
                
        def safe_float(val):
            try:
                return float(val)
            except (ValueError, TypeError):
                return 0.0

        post_count = safe_int(row["post_count"])
        viral_count = safe_int(row["viral_count"])
        total_score = safe_float(row["total_score"])
        max_score = safe_float(row["max_score"])
        last_time = safe_int(row["last_time"])
        
        stats[key] = {
            f"{prefix}_prior_posts": post_count,
            f"{prefix}_prior_viral_count": viral_count,
            f"{prefix}_prior_mean_score": total_score / post_count if post_count > 0 else 0.0,
            f"{prefix}_prior_viral_rate": viral_count / post_count if post_count > 0 else 0.0,
            f"{prefix}_prior_max_score": max_score,
            f"{prefix}_last_time": last_time,
        }
        if top_flag_col:
             stats[key][f"{prefix}_prior_top1pct_count"] = safe_int(row["top_flag_count"])
             
    return stats

def add_historical_features_from_store(
    df: pd.DataFrame,
    stats_store: Dict[str, Any] | None,
    group_col: str | Sequence[str],
    prefix: str,
    top_flag_col: str | None = None,
    top_flag_output: str | None = None,
) -> pd.DataFrame:
    store_key = prefix
    store_data = stats_store.get(store_key, {}) if stats_store else {}
    global_stats = stats_store.get("global", {}) if stats_store else {}
    
    stats_rows = []
    group_cols = [group_col] if isinstance(group_col, str) else list(group_col)
    
    for _, row in df.iterrows():
        if len(group_cols) == 1:
            key = str(row[group_cols[0]])
        else:
            key = "|".join(str(row[c]) for c in group_cols)
            
        entry = store_data.get(key)
        if not entry:
            entry = {
                f"{prefix}_prior_posts": 0,
                f"{prefix}_prior_viral_count": 0,
                f"{prefix}_prior_mean_score": global_stats.get("overall_score", 0.0),
                f"{prefix}_prior_viral_rate": global_stats.get("overall_viral_rate", 0.0),
                f"{prefix}_prior_max_score": global_stats.get("overall_score", 0.0),
                f"{prefix}_last_time": 0,
            }
            if top_flag_output:
                entry[top_flag_output] = 0
        
        current_time = row["time"]
        last_time = entry.get(f"{prefix}_last_time", 0)
        if last_time > 0:
            hours_since = (current_time - last_time) / 3600.0
        else:
            hours_since = global_stats.get("median_hours_since_last", 24.0)
            
        row_stats = entry.copy()
        row_stats[f"{prefix}_hours_since_last"] = hours_since
        stats_rows.append(row_stats)
        
    stats_df = pd.DataFrame(stats_rows, index=df.index)
    
    stats_df[f"{prefix}_prior_posts"] = stats_df[f"{prefix}_prior_posts"].astype(np.float32)
    stats_df[f"{prefix}_prior_posts_log1p"] = np.log1p(stats_df[f"{prefix}_prior_posts"]).astype(np.float32)
    stats_df[f"{prefix}_prior_viral_count"] = stats_df[f"{prefix}_prior_viral_count"].astype(np.float32)
    stats_df[f"{prefix}_prior_mean_score"] = stats_df[f"{prefix}_prior_mean_score"].astype(np.float32)
    stats_df[f"{prefix}_prior_viral_rate"] = stats_df[f"{prefix}_prior_viral_rate"].astype(np.float32)
    stats_df[f"{prefix}_hours_since_last"] = stats_df[f"{prefix}_hours_since_last"].astype(np.float32)
    stats_df[f"{prefix}_prior_max_score"] = stats_df[f"{prefix}_prior_max_score"].astype(np.float32)
    
    if top_flag_output and top_flag_output in stats_df.columns:
            stats_df[top_flag_output] = stats_df[top_flag_output].astype(np.float32)

    if prefix in {"user", "user_domain"}:
        stats_df[f"{prefix}_avg_score"] = stats_df[f"{prefix}_prior_mean_score"]
        stats_df[f"{prefix}_mean_score"] = stats_df[f"{prefix}_prior_mean_score"]
        stats_df[f"{prefix}_max_score"] = stats_df[f"{prefix}_prior_max_score"]
        
    return df.join(stats_df)

def export_stats(data_path: Path, output_path: Path):
    # Import here to avoid circular dependency
    from src.features import load_raw_posts, FeatureEngineer

    print("Loading data...")
    df = load_raw_posts(data_path)
    
    # Clean up data
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0)
    
    # Replace infs
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    df["time"] = df["time"].astype(int)
    df["score"] = df["score"].astype(float)
    df["url"] = df["url"].fillna("")
    
    # Pre-process using FeatureEngineer logic (for domain extraction, etc)
    fe = FeatureEngineer()
    # We only need partial transform to get 'domain', 'is_viral', 'is_top_1pct'
    df["is_viral"] = (df["score"] > fe.viral_threshold).astype(np.int8)
    df["is_top_1pct"] = fe._mark_top_fraction(df["score"], 0.01)
    df = fe._add_url_features(df) # Adds 'domain'
    
    print("Calculating stats...")
    
    user_stats = calculate_stats(df, "by", "user", top_flag_col="is_top_1pct")
    domain_stats = calculate_stats(df, "domain", "domain")
    user_domain_stats = calculate_stats(df, ["by", "domain"], "user_domain")
    
    # Global stats
    global_stats = {
        "overall_viral_rate": float(df["is_viral"].mean()),
        "overall_score": float(df["score"].mean()),
        # Median hours since last is hard to pre-calc perfectly without full diffs, 
        # but we can use a heuristic or just the value from training (24.0)
        "median_hours_since_last": 24.0 
    }
    
    output = {
        "user": user_stats,
        "domain": domain_stats,
        "user_domain": user_domain_stats,
        "global": global_stats
    }
    
    print(f"Saving stats to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(output, f)
        
    print("Done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default="data/hn_posts.csv")
    parser.add_argument("--output-path", type=Path, default="reports/feature_stats.json")
    args = parser.parse_args()
    
    export_stats(args.data_path, args.output_path)

if __name__ == "__main__":
    main()
