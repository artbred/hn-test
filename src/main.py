import argparse
import json
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

warnings.filterwarnings("ignore")


def get_device():
    """Detect and return the best available device (MPS for M chip, CUDA for NVIDIA, CPU otherwise)"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using Apple Silicon (MPS) for training")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("✓ Using CUDA (NVIDIA GPU) for training")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU for training")
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
        use_user_features=False,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.usernames = usernames
        self.user_stats = user_stats
        self.use_user_features = use_user_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Prepend user features if enabled
        if self.use_user_features and self.user_stats is not None:
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


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for addressing class imbalance in binary/multi-class classification.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    https://arxiv.org/abs/1708.02002

    Args:
        alpha: Weighting factor for positive class (range: 0-1). Higher values give more
               weight to minority class. Default: 0.75
        gamma: Focusing parameter for modulating loss (range: 0-5). Higher values focus
               more on hard examples. Default: 2.0
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (batch_size, num_classes)
            targets: Ground truth labels (batch_size)

        Returns:
            Focal loss value
        """
        # Calculate cross entropy loss (no reduction to apply focal modulation per sample)
        ce_loss = torch.nn.functional.cross_entropy(
            inputs, targets, reduction="none"
        )

        # Get probabilities from logits
        pt = torch.exp(-ce_loss)

        # Apply focal loss formula: FL = -alpha_t * (1 - pt)^gamma * log(pt)
        # For binary classification with targets 0 or 1:
        # - alpha_t = alpha if target=1 (minority/positive class)
        # - alpha_t = (1-alpha) if target=0 (majority/negative class)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Calculate focal loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


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


def compute_focal_loss_func(outputs, labels, num_items_in_batch, focal_loss):
    """
    Custom loss function for Trainer using Focal Loss.

    Args:
        outputs: Model outputs (ModelOutput object or dict with 'logits')
        labels: Ground truth labels
        num_items_in_batch: Number of items in accumulated batch
        focal_loss: FocalLoss instance to use for computing loss

    Returns:
        loss: Computed focal loss value
    """
    logits = (
        outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
    )
    loss = focal_loss(logits, labels)
    return loss


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
    parser.add_argument(
        "--focal-alpha",
        type=float,
        default=0.75,
        help="Focal loss alpha parameter: weight for positive class (0-1). Higher values favor minority class. (default: 0.75)",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma parameter: focusing parameter (0-5). Higher values focus more on hard examples. (default: 2.0)",
    )
    parser.add_argument(
        "--use-user-features",
        action="store_true",
        default=True,
        help="Enable user performance features (viral rate, avg score, post count, avg comments)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for tokenization. Use 200+ when using user features. (default: 128)",
    )
    args = parser.parse_args()

    # Detect device (MPS for M chip, CUDA for NVIDIA, CPU otherwise)
    device = get_device()

    # Load your data
    print(f"\nLoading data from {args.data_path}...")
    df = load_jsonl(args.data_path)

    df["is_viral"] = (df["score"] > args.viral_threshold).astype(int)

    print(df[["title", "score"]].isnull().sum())

    df = df.dropna(subset=["title", "score"])
    print(f"Dataset shape after removing NaN: {df.shape}")

    # Clean titles - remove any weird characters or excessive whitespace
    df["title"] = df["title"].astype(str).apply(lambda x: " ".join(x.split()))

    # Remove empty or very short titles (less than 3 characters)
    df = df[df["title"].str.len() > 3]
    print(f"Dataset shape after removing short titles: {df.shape}")

    # Limit samples if specified
    if args.max_samples is not None:
        print(f"\n⚠ Limiting dataset to {args.max_samples} samples")
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42)
        print(f"Dataset shape after limiting: {df.shape}")

    # Check class distribution
    print("\n=== Class Distribution ===")
    print(df["is_viral"].value_counts())
    print("\nPercentage distribution:")
    print(df["is_viral"].value_counts(normalize=True) * 100)

    print(f"\nTotal samples: {len(df)}")
    print(
        f"Viral posts: {df['is_viral'].sum()} ({df['is_viral'].sum() / len(df) * 100:.2f}%)"
    )
    print(
        f"Non-viral posts: {(~df['is_viral'].astype(bool)).sum()} ({(~df['is_viral'].astype(bool)).sum() / len(df) * 100:.2f}%)"
    )

    # Split the data FIRST to prevent data leakage
    # Important: We must compute user statistics ONLY on training data
    print("\n=== Splitting Data ===")
    train_indices, test_indices = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=df["is_viral"],
    )

    # Create train/test dataframes
    train_df = df.iloc[train_indices].copy()
    test_df = df.iloc[test_indices].copy()

    print(
        f"Training samples: {len(train_df)} ({len(train_df) / len(df) * 100:.1f}%)"
    )
    print(f"Test samples: {len(test_df)} ({len(test_df) / len(df) * 100:.1f}%)")

    # Compute user statistics ONLY on training data to prevent data leakage
    user_stats = None
    if args.use_user_features:
        # Validate that "by" column exists
        if "by" not in train_df.columns:
            print(
                "\n⚠ Error: 'by' column not found in data. User features require usernames."
            )
            print(
                "Disabling user features and continuing with text-only model..."
            )
            args.use_user_features = False
        else:
            print("\n=== Computing User Statistics (Training Data Only) ===")
            user_stats = compute_user_statistics(
                train_df, viral_threshold=args.viral_threshold
            )
            print(
                f"✓ Computed statistics for {len(user_stats) - 1} unique users from training data"
            )

            # Display some example user stats
            sample_users = list(user_stats.keys())[:3]
            for username in sample_users:
                if username != "__DEFAULT__":
                    stats = user_stats[username]
                    print(
                        f"  {username}: viral_rate={stats['viral_rate']:.2%}, "
                        f"avg_score={stats['avg_score']:.0f}, posts={stats['post_count']}"
                    )

            # Check for cold-start users in test set
            test_users = set(test_df["by"].dropna().unique())
            train_users = set(train_df["by"].dropna().unique())
            cold_start_users = test_users - train_users
            if cold_start_users:
                # Prevent division by zero
                percentage = (
                    (len(cold_start_users) / len(test_users) * 100)
                    if len(test_users) > 0
                    else 0.0
                )
                print(
                    f"\n⚠ Cold-start users in test set: {len(cold_start_users)} "
                    f"({percentage:.1f}% of test users)"
                )
                print(
                    "  These users will use default statistics (median values from training data)"
                )

    # Extract features from train/test dataframes
    X_train = train_df["title"].values
    y_train = train_df["is_viral"].values
    X_test = test_df["title"].values
    y_test = test_df["is_viral"].values

    if args.use_user_features:
        usernames_train = train_df["by"].values
        usernames_test = test_df["by"].values
    else:
        usernames_train = None
        usernames_test = None

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
        max_length=args.max_length,
        usernames=usernames_train,
        user_stats=user_stats,
        use_user_features=args.use_user_features,
    )
    test_dataset = TitleDataset(
        X_test,
        y_test,
        tokenizer,
        max_length=args.max_length,
        usernames=usernames_test,
        user_stats=user_stats,
        use_user_features=args.use_user_features,
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
        save_strategy="epoch",
        eval_strategy="epoch",  # Evaluate at end of each epoch
        load_best_model_at_end=True,  # Load the best model when finished
        metric_for_best_model="f1",  # Use F1 score to determine best model
        greater_is_better=True,  # Higher F1 is better
        # Use fp16 for better performance on Apple Silicon
        fp16=device.type == "cuda",  # Only for NVIDIA GPUs
        # For MPS (Apple Silicon), use default precision as fp16 is not stable yet
        use_cpu=False,  # Ensure we use GPU acceleration
    )

    # Create focal loss instance
    focal_loss = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    print(
        f"\n✓ Using Focal Loss with alpha={args.focal_alpha}, gamma={args.focal_gamma}"
    )

    # Train with custom loss function and automatic metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_loss_func=lambda outputs,
        labels,
        num_items_in_batch: compute_focal_loss_func(
            outputs, labels, num_items_in_batch, focal_loss
        ),
        compute_metrics=compute_metrics,
    )

    print("\n=== Starting Training ===")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Max sequence length: {args.max_length}")
    print(
        f"User features: {'Enabled' if args.use_user_features else 'Disabled'}"
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    trainer.train()

    # Evaluate the model (metrics computed automatically via compute_metrics)
    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    metrics = trainer.evaluate()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print("=" * 60)

    # Save the model
    print("\n=== Saving Model ===")
    trainer.save_model("./results/final_model")
    tokenizer.save_pretrained("./results/final_model")
    print("Model saved to ./results/final_model")

    # Save metrics to file
    # Filter to save only the core metrics (excluding runtime/samples_per_second etc)
    metrics_to_save = {
        key: float(value) if isinstance(value, (int, float)) else value
        for key, value in metrics.items()
        if key
        in [
            "eval_loss",
            "eval_accuracy",
            "eval_precision",
            "eval_recall",
            "eval_f1",
        ]
    }

    with open("./results/evaluation_metrics.json", "w") as f:
        json.dump(metrics_to_save, f, indent=2)
    print("Evaluation metrics saved to ./results/evaluation_metrics.json")


if __name__ == "__main__":
    main()
