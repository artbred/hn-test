import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
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
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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


def evaluate_model(trainer, test_dataset):
    """
    Evaluate the trained model and display comprehensive metrics including confusion matrix
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    # Get predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    true_labels = predictions.label_ids

    # Calculate metrics (with zero_division parameter to avoid warnings)
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="binary")
    recall = recall_score(true_labels, pred_labels, average="binary")
    f1 = f1_score(true_labels, pred_labels, average="binary")

    # Display overall metrics
    print("\n--- Overall Metrics ---")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Display detailed classification report
    print("\n--- Detailed Classification Report ---")
    class_names = ["Non-Viral (0)", "Viral (1)"]
    print(
        classification_report(
            true_labels, pred_labels, target_names=class_names, digits=4
        )
    )

    # Calculate and display confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("\n--- Confusion Matrix ---")
    print("Format: [True Label] vs [Predicted Label]")
    print()
    print("                 Predicted Non-Viral  Predicted Viral")
    print(f"Actual Non-Viral      {cm[0][0]:6d}              {cm[0][1]:6d}")
    print(f"Actual Viral          {cm[1][0]:6d}              {cm[1][1]:6d}")
    print()

    # Explain confusion matrix components
    tn, fp, fn, tp = cm.ravel()
    print("Breakdown:")
    print(f"  True Negatives (TN):  {tn:6d} - Correctly predicted as Non-Viral")
    print(f"  False Positives (FP): {fp:6d} - Incorrectly predicted as Viral")
    print(
        f"  False Negatives (FN): {fn:6d} - Incorrectly predicted as Non-Viral"
    )
    print(f"  True Positives (TP):  {tp:6d} - Correctly predicted as Viral")
    print()

    # Calculate additional insights
    total = len(true_labels)
    correct = tn + tp
    incorrect = fp + fn

    print("--- Summary ---")
    print(f"Total predictions: {total}")
    print(f"Correct predictions: {correct} ({correct / total * 100:.2f}%)")
    print(
        f"Incorrect predictions: {incorrect} ({incorrect / total * 100:.2f}%)"
    )
    print()

    # Class-specific accuracy
    non_viral_accuracy = tn / (tn + fp) if (tn + fp) > 0 else 0
    viral_accuracy = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("Class-specific Performance:")
    print(
        f"  Non-Viral detection rate: {non_viral_accuracy * 100:.2f}% ({tn}/{tn + fn})"
    )
    print(
        f"  Viral detection rate:     {viral_accuracy * 100:.2f}% ({tp}/{tp + fn})"
    )

    print("=" * 60)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
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

    X = df["title"].values
    y = df["is_viral"].values

    print(f"\nTotal samples: {len(X)}")
    print(f"Viral posts: {sum(y)} ({sum(y) / len(y) * 100:.2f}%)")
    print(
        f"Non-viral posts: {len(y) - sum(y)} ({(len(y) - sum(y)) / len(y) * 100:.2f}%)"
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model_name = "bert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    )

    # Create datasets
    train_dataset = TitleDataset(X_train, y_train, tokenizer)
    test_dataset = TitleDataset(X_test, y_test, tokenizer)

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
        # Use fp16 for better performance on Apple Silicon
        fp16=device.type == "cuda",  # Only for NVIDIA GPUs
        # For MPS (Apple Silicon), use default precision as fp16 is not stable yet
        use_cpu=False,  # Ensure we use GPU acceleration
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
    )

    print("\n=== Starting Training ===")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    trainer.train()

    # Evaluate the model
    metrics = evaluate_model(trainer, test_dataset)

    # Save the model
    print("\n=== Saving Model ===")
    trainer.save_model("./results/final_model")
    tokenizer.save_pretrained("./results/final_model")
    print("Model saved to ./results/final_model")

    # Save metrics to file
    metrics_dict = {
        "accuracy": float(metrics["accuracy"]),
        "precision": float(metrics["precision"]),
        "recall": float(metrics["recall"]),
        "f1": float(metrics["f1"]),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }

    with open("./results/evaluation_metrics.json", "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print("Evaluation metrics saved to ./results/evaluation_metrics.json")


if __name__ == "__main__":
    main()
