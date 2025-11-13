import argparse
import json
import logging
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def log_dataset_summary(name, df, label_col="is_viral"):
    if df.empty:
        logger.info("%s: n=0", name)
        return
    total = len(df)
    viral = int(df[label_col].sum())
    non_viral = total - viral
    logger.info(
        "%s: n=%d | viral=%d (%.1f%%) | non-viral=%d (%.1f%%)",
        name,
        total,
        viral,
        (viral / total) * 100,
        non_viral,
        (non_viral / total) * 100,
    )


def log_training_configuration(device, batch_size, epochs, max_length, sizes):
    train_size, val_size, test_size = sizes
    logger.info(
        "Training config | device=%s | batch=%d | epochs=%d | max_len=%d",
        device,
        batch_size,
        epochs,
        max_length,
    )
    logger.info(
        "Dataset sizes | train=%d | val=%d | test=%d",
        train_size,
        val_size,
        test_size,
    )


def get_device():
    """Detect and return the best available device (MPS for M chip, CUDA for NVIDIA, CPU otherwise)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Detected device: Apple Silicon (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Detected device: CUDA (NVIDIA GPU)")
    else:
        device = torch.device("cpu")
        logger.info("Detected device: CPU")
    return device


class TitleDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        tokenizer,
        max_length=128,
        usernames=None,
        user_stats=None,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.usernames = usernames
        self.user_stats = user_stats

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Prepend user features if enabled
        if self.user_stats is not None:
            username = (
                self.usernames[idx] if self.usernames is not None else None
            )
            # Explicitly check for NaN to avoid dictionary key issues
            if username is not None and not pd.isna(username):
                user_feature_text = generate_user_feature_text(
                    username, self.user_stats
                )
                text = user_feature_text + text

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# 1. Load JSONL file
def load_jsonl(file_path):
    """Load data from JSONL file"""
    data = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)


# Basic preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = " ".join(text.split())
    return text


# Feature engineering
def extract_features(text):
    # Extract features from text
    features = []
    for word in text.split():
        features.append(len(word))
    return features


def compute_user_statistics(df, viral_threshold=500):
    """
    Compute per-user statistics for feature engineering.

    Args:
        df: DataFrame with columns 'by' (username), 'score', 'descendants'
        viral_threshold: Score threshold for viral classification

    Returns:
        Dictionary mapping username -> user statistics dict
    """
    user_stats = {}

    # Always create default stats first (fallback for edge cases)
    user_stats["__DEFAULT__"] = {
        "viral_rate": 0.0,
        "avg_score": 0.0,
        "post_count": 1.0,
        "avg_comments": 0.0,
    }

    # Group by user
    for username, user_df in df.groupby("by"):
        if pd.isna(username):
            continue

        # Calculate statistics
        scores = user_df["score"].values
        viral_posts = (scores > viral_threshold).sum()
        total_posts = len(user_df)

        stats = {
            "viral_rate": viral_posts / total_posts if total_posts > 0 else 0.0,
            "avg_score": float(np.mean(scores)) if len(scores) > 0 else 0.0,
            "post_count": total_posts,
            "avg_comments": (
                float(np.mean(user_df["descendants"].fillna(0)))
                if "descendants" in user_df.columns
                else 0.0
            ),
        }

        user_stats[username] = stats

    # Update default stats with actual medians if we have valid users
    if len(user_stats) > 1:  # More than just __DEFAULT__
        # Collect stats from actual users (excluding __DEFAULT__)
        all_viral_rates = [
            s["viral_rate"] for k, s in user_stats.items() if k != "__DEFAULT__"
        ]
        all_avg_scores = [
            s["avg_score"] for k, s in user_stats.items() if k != "__DEFAULT__"
        ]
        all_post_counts = [
            s["post_count"] for k, s in user_stats.items() if k != "__DEFAULT__"
        ]
        all_avg_comments = [
            s["avg_comments"]
            for k, s in user_stats.items()
            if k != "__DEFAULT__"
        ]

        # Update default with medians
        user_stats["__DEFAULT__"] = {
            "viral_rate": (
                float(np.median(all_viral_rates)) if all_viral_rates else 0.0
            ),
            "avg_score": float(np.median(all_avg_scores))
            if all_avg_scores
            else 0.0,
            "post_count": (
                float(np.median(all_post_counts)) if all_post_counts else 1.0
            ),
            "avg_comments": (
                float(np.median(all_avg_comments)) if all_avg_comments else 0.0
            ),
        }

    return user_stats


def generate_user_feature_text(username, user_stats):
    """
    Convert user statistics to natural language text for BERT input.

    Args:
        username: Username to lookup
        user_stats: Dictionary of user statistics from compute_user_statistics

    Returns:
        Natural language string describing user's posting history
    """
    # Get stats for this user, or use default for unknown users
    stats = user_stats.get(username, user_stats["__DEFAULT__"])

    # Convert to percentages and round for readability
    viral_rate_pct = int(stats["viral_rate"] * 100)
    avg_score = int(stats["avg_score"])
    post_count = int(stats["post_count"])
    avg_comments = int(stats["avg_comments"])

    # Generate natural language description
    feature_text = (
        f"User has {viral_rate_pct}% viral rate, "
        f"average score {avg_score}, "
        f"{post_count} posts, "
        f"{avg_comments} avg comments. "
    )

    return feature_text


def compute_metrics(eval_pred):
    """
    Compute metrics for automatic evaluation during training.

    This function is called by Trainer during evaluation to compute metrics.

    Args:
        eval_pred: EvalPrediction object with predictions and label_ids

    Returns:
        Dictionary of metric names to values
    """
    predictions = np.argmax(eval_pred.predictions, axis=1)
    labels = eval_pred.label_ids

    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(
        labels, predictions, average="binary", zero_division=0
    )
    recall = recall_score(
        labels, predictions, average="binary", zero_division=0
    )
    f1 = f1_score(labels, predictions, average="binary", zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train HackerNews virality prediction model"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/hn_posts.jsonl",
        help="Path to the JSONL data file",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (useful for testing). If not specified, uses all data.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Training batch size (default: 8, optimized for M chip)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--viral-threshold",
        type=int,
        default=500,
        help="Score threshold for viral classification (default: 500)",
    )
    args = parser.parse_args()

    MAX_LENGTH = 256
    VAL_SIZE = 0.1
    TEST_SIZE = 0.2
    SEED = 42
    
    device = get_device()

    # Load your data
    logger.info("Loading data from %s", args.data_path)
    df = load_jsonl(args.data_path)

    df["is_viral"] = (df["score"] > args.viral_threshold).astype(int)

    null_counts = df[["title", "score"]].isnull().sum().to_dict()
    if any(null_counts.values()):
        logger.info("Null counts (title/score): %s", null_counts)

    df = df.dropna(subset=["title", "score"])
    logger.info("Dataset shape after removing NaNs: %s", df.shape)

    df["title"] = df["title"].astype(str).apply(lambda x: " ".join(x.split()))

    # Remove empty or very short titles (less than 3 characters)
    df = df[df["title"].str.len() > 3]
    logger.info("Dataset shape after removing short titles: %s", df.shape)

    # Limit samples if specified
    if args.max_samples is not None:
        logger.warning("Limiting dataset to %d samples", args.max_samples)
        df = df.sample(n=min(args.max_samples, len(df)), random_state=SEED)
        logger.info("Dataset shape after limiting: %s", df.shape)

    log_dataset_summary("Dataset summary", df)

    # Split the data FIRST to prevent data leakage (stats only from training set)
    logger.info(
        "Creating stratified splits | val=%.0f%% | test=%.0f%%",
        VAL_SIZE * 100,
        TEST_SIZE * 100,
    )

    all_indices = np.arange(len(df))
    train_val_indices, test_indices = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df["is_viral"],
    )

    val_relative_size = VAL_SIZE / (1.0 - TEST_SIZE)
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_relative_size,
        random_state=SEED,
        stratify=df.iloc[train_val_indices]["is_viral"],
    )
    

    # Create train/val/test dataframes
    train_df = df.iloc[train_indices].copy()
    val_df = (
        df.iloc[val_indices].copy() if len(val_indices) else pd.DataFrame(columns=df.columns)
    )
    test_df = df.iloc[test_indices].copy()

    log_dataset_summary("Train split", train_df)
    if len(val_df):
        log_dataset_summary("Validation split", val_df)
    else:
        logger.info("Validation split disabled")
    log_dataset_summary("Test split", test_df)

    user_training_stats = compute_user_statistics(
        train_df, viral_threshold=args.viral_threshold
    )

    logger.info(
        "Computed user statistics for %d training users",
        max(len(user_training_stats) - 1, 0),
    )

    # Extract features from train/val/test dataframes
    X_train = train_df["title"].values
    y_train = train_df["is_viral"].values
    X_val = val_df["title"].values if len(val_df) else np.array([])
    y_val = val_df["is_viral"].values if len(val_df) else np.array([])
    X_test = test_df["title"].values
    y_test = test_df["is_viral"].values

    usernames_train = train_df["by"].values
    usernames_val = val_df["by"].values if len(val_df) else None
    usernames_test = test_df["by"].values

    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # Create datasets
    train_dataset = TitleDataset(
        X_train,
        y_train,
        tokenizer,
        max_length=MAX_LENGTH,
        usernames=usernames_train,
        user_stats=user_training_stats,
    )
    val_dataset = (
        TitleDataset(
            X_val,
            y_val,
            tokenizer,
            max_length=MAX_LENGTH,
            usernames=usernames_val,
            user_stats=user_training_stats,
        )
        if len(val_df)
        else None
    )
    test_dataset = TitleDataset(
        X_test,
        y_test,
        tokenizer,
        max_length=MAX_LENGTH,
        usernames=usernames_test,
        user_stats=user_training_stats,
    )

    # Training arguments optimized for Apple Silicon
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch"
    )

    # Train with custom loss function and automatic metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if val_dataset is not None else test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    log_training_configuration(
        device,
        args.batch_size,
        args.epochs,
        MAX_LENGTH,
        (
            len(train_dataset),
            len(val_dataset) if val_dataset is not None else 0,
            len(test_dataset),
        ),
    )

    trainer.train()

    logger.info("Evaluating on held-out test set")
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    for metric_name, metric_value in test_metrics.items():
        if isinstance(metric_value, (int, float, np.floating, np.integer)):
            logger.info("%s: %.4f", metric_name, metric_value)
        else:
            logger.info("%s: %s", metric_name, metric_value)

    # Save the model
    logger.info("Saving model artifacts to ./results/final_model")
    trainer.save_model("./results/final_model")
    tokenizer.save_pretrained("./results/final_model")
    logger.info("Model saved to ./results/final_model")



if __name__ == "__main__":
    main()
