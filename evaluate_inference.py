
import os
import pandas as pd
import numpy as np
from src.features import load_raw_posts
from predict import Predictor

# Configuration
MODEL_PATH = "reports/catboost_model.cbm"
STATS_PATH = "reports/feature_stats.json"
VIRAL_THRESHOLD = 250  # Same as in training

def main():
    print("Loading data...")
    try:
        df = load_raw_posts("data/hn_posts.csv")
    except FileNotFoundError:
        print("Error: 'data/hn_posts.csv' not found.")
        return

    # Data cleaning
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["time"] = pd.to_numeric(df["time"], errors="coerce").fillna(0).astype(int)
    df["title"] = df["title"].fillna("")
    df["url"] = df["url"].fillna("")
    df["by"] = df["by"].fillna("unknown")

    print(f"Total posts loaded: {len(df)}")

    # --- Smart Sampling Strategy ---
    # We want to see how the model distinguishes between:
    # 1. High Score (Viral)
    # 2. Mid Score (Popular but not viral)
    # 3. Low Score (Ignored)
    
    samples = []
    
    # 1. Top Viral: Highest scoring posts
    top_viral = df.nlargest(200, "score")
    top_viral["sample_type"] = "High Viral"
    samples.append(top_viral)
    
    # 2. Mid Range: Score between 50 and 100
    mid_range = df[(df["score"] >= 50) & (df["score"] <= 100)]
    if not mid_range.empty:
        mid_sample = mid_range.sample(min(200, len(mid_range)))
        mid_sample["sample_type"] = "Mid Range"
        samples.append(mid_sample)
        
    # 3. Low Range: Score <= 2
    low_range = df[df["score"] <= 2]
    if not low_range.empty:
        low_sample = low_range.sample(min(200, len(low_range)))
        low_sample["sample_type"] = "Low Score"
        samples.append(low_sample)

    # 4. Recent: To check if time affects it (optional, but let's stick to score for now)
    
    eval_df = pd.concat(samples).reset_index(drop=True)
    print(f"Selected {len(eval_df)} posts for evaluation.")

    # --- Setup Predictor ---
    print("Initializing Predictor...")
    os.environ["CATBOOST_MODEL_PATH"] = MODEL_PATH
    os.environ["FEATURE_STATS_PATH"] = STATS_PATH
    
    predictor = Predictor()
    try:
        predictor.setup()
    except Exception as e:
        print(f"Error setting up predictor: {e}")
        print("Ensure you have trained the model and generated stats (run `src/train.py`).")
        return

    # --- Run Inference ---
    print("\nRunning Inference...")
    results = []
    
    for _, row in eval_df.iterrows():
        try:
            pred = predictor.predict(
                title=row["title"],
                url=row["url"],
                by=row["by"],
                time=row["time"]
            )
            prob = pred.get("probability")
            
            results.append({
                "Type": row["sample_type"],
                "Score": row["score"],
                "Predicted_Prob": round(prob, 4) if prob is not None else 0.0,
                "Title": row["title"][:60] + "..." if len(row["title"]) > 60 else row["title"],
                "Author": row["by"]
            })
        except Exception as e:
            print(f"Prediction failed for post {row['id']}: {e}")

    # --- Display Results ---
    res_df = pd.DataFrame(results)
    
    # Sort by Predicted Probability to see alignment
    res_df = res_df.sort_values("Predicted_Prob", ascending=False)
    
    print("\n" + "="*80)
    print("INFERENCE EVALUATION RESULTS")
    print("="*80)
    
    # Format for nice printing
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 70)
    
    print(res_df[["Type", "Score", "Predicted_Prob", "Title", "Author"]].to_string(index=False))
    
    print("\n" + "="*80)
    
    # Quick Analysis
    print("Analysis:")
    high_conf = res_df[res_df["Predicted_Prob"] > 0.5]
    print(f"Posts predicted viral (>0.5): {len(high_conf)}")
    if not high_conf.empty:
        true_positives = len(high_conf[high_conf["Score"] >= VIRAL_THRESHOLD])
        print(f"  - Actually viral (Score >= {VIRAL_THRESHOLD}): {true_positives}")
        print(f"  - False positives: {len(high_conf) - true_positives}")

if __name__ == "__main__":
    main()
