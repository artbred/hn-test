#!/usr/bin/env python3
"""
Convert JSONL data to CSV format for model training.
"""
import json
import sys
from pathlib import Path
import pandas as pd


def convert_jsonl_to_csv(jsonl_path: str, csv_path: str) -> None:
    """Convert JSONL file to CSV format."""
    # Read JSONL file
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        print("Error: No data found in JSONL file", file=sys.stderr)
        sys.exit(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure required columns exist
    required_cols = ['id', 'title', 'url', 'type', 'by', 'time', 'score']
    for col in required_cols:
        if col not in df.columns:
            if col == 'text':
                df[col] = ''
            elif col == 'type':
                df[col] = 'story'
            else:
                df[col] = 0
    
    # Save as CSV
    df.to_csv(csv_path, index=False)
    print(f"Converted {len(df):,} records from JSONL to CSV")
    print(f"Output: {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_jsonl_to_csv.py <input.jsonl> <output.csv>")
        sys.exit(1)
    
    jsonl_file = sys.argv[1]
    csv_file = sys.argv[2]
    
    if not Path(jsonl_file).exists():
        print(f"Error: Input file not found: {jsonl_file}", file=sys.stderr)
        sys.exit(1)
    
    convert_jsonl_to_csv(jsonl_file, csv_file)
