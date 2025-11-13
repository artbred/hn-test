# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**HN Virality Predictor**: A machine learning system that predicts whether Hacker News post titles will go viral (exceed a configurable score threshold, default: 500 points) using fine-tuned BERT with optional user performance features.

**Architecture**: Single-file monolithic design in `src/main.py` (643 lines). The pipeline: JSONL data → Preprocessing → BERT fine-tuning → Binary classification (viral/non-viral).

**Two Operation Modes**:
1. **Text-only mode**: Classifies based solely on post titles
2. **Text + User features mode** (default): Prepends natural language user statistics to titles before tokenization

## Common Commands

### Environment Setup
```bash
# Python 3.9 required (see .python-version)
uv sync  # Install dependencies using uv package manager
# OR
pip install -e .
```

### Training Commands
```bash
# Basic training with user features (default)
python src/main.py --data-path data/hn_posts.jsonl

# Full configuration example
python src/main.py \
  --data-path data/hn_posts.jsonl \
  --batch-size 8 \
  --epochs 3 \
  --viral-threshold 500 \
  --use-user-features \
  --max-length 256

# Quick test with limited samples
python src/main.py --max-samples 1000 --epochs 1

# Text-only mode (no user features)
python src/main.py --no-use-user-features --max-length 128

# Adjust focal loss parameters for class imbalance
python src/main.py --focal-alpha 0.75 --focal-gamma 2.0
```

**Note**: No test suite or linting configuration currently exists in the project.

## Key Architecture Patterns

### Data Leakage Prevention (CRITICAL)
User statistics are computed **only** on the training set to prevent data leakage:

```python
# src/main.py:177-287
train_df, test_df = train_test_split(...)
user_stats = compute_user_statistics(train_df)  # NOT full df!
# Test users not in training data receive median defaults
```

This is essential for valid model evaluation. Any changes to user statistics computation must maintain this separation.

### Natural Language Feature Injection
User performance features are converted to natural language and prepended to titles:
- Format: `"User has 45% viral rate, average score 320, 12 posts, 8 avg comments. [ORIGINAL_TITLE]"`
- Allows BERT to learn user patterns without architectural changes
- Enabled by default with `--use-user-features` flag

### Focal Loss for Class Imbalance
Custom focal loss implementation (src/main.py:93-147) addresses severely imbalanced data:
- **Alpha** (default 0.75): Weight for minority viral class
- **Gamma** (default 2.0): Focusing parameter for hard examples
- Formula: `FL = -alpha_t * (1 - pt)^gamma * log(pt)`

### Hardware Flexibility
Automatic device detection with priority: Apple Silicon MPS → NVIDIA CUDA → CPU fallback.

## Code Structure

```
src/main.py structure:
├── TitleDataset (lines 40-91)          # Custom dataset with optional user features
├── FocalLoss (lines 93-147)            # Class imbalance handling
├── load_data (lines 149-175)           # JSONL loading and preprocessing
├── compute_user_statistics (lines 177-287)  # User feature engineering
├── compute_metrics (lines 289-340)     # Evaluation metrics
└── main (lines 342-642)                # Training pipeline

results/ directory:
├── checkpoint-*/                        # Model checkpoints per epoch
├── final_model/                        # Best model by F1 score
└── evaluation_metrics.json             # Test set performance
```

## Data Format

JSONL (JSON Lines) format required with fields:
- `id`: Post ID
- `title`: Post title (text input)
- `score`: Post score (label generation)
- `by`: Username (user statistics)
- `type`: Post type
- `descendants`: Number of comments
- `time`: Unix timestamp

## Known Issues and TODOs

### Current Model Performance Issue
The model is experiencing collapse - predicting all posts as non-viral (96.97% accuracy but 0% precision/recall/F1). This indicates:
- The class imbalance is too severe for current approach
- Focal loss parameters may need tuning
- Consider data augmentation, ensemble methods, or different viral threshold

### TODO Items (from TODO.txt)
- Save metrics as pickle for loading during inference
- Evaluate whether focal loss should be removed
- Add more user-based metrics beyond viral rate/avg score/post count

## Key Technologies

- **Python 3.9** with `uv` package manager
- **PyTorch 2.8.0** for deep learning
- **HuggingFace Transformers 4.57.1** with `bert-base-uncased`
- **Pandas 2.3.3**, NumPy 2.0.2 for data processing
- **scikit-learn 1.6.1** for train/test split and metrics

## Development Notes

- Training uses stratified 80/20 train/test split
- Model selection based on best F1 score across epochs
- Handles cold-start for unknown users with median statistics
- No inline API documentation beyond basic docstrings
- `src/main.py` currently has uncommitted changes (git status: modified)
