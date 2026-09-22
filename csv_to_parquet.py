"""
Sample a large Reddit CSV, keep only the columns needed for the
stance-scoring prototype, and save as parquet.

Usage:
    python3 csv_to_parquet.py input.csv output.parquet --n 5000 --mode random
"""

import argparse
import pandas as pd

# Columns we actually need for the stance-scoring + event-response pipeline.
# Anything not in this list gets dropped (author karma, downs, awards, etc.)
KEEP_COLS = [
    "comment_id",
    "self_text",
    "subreddit",
    "created_time",
    "post_id",
    "score",
    "controversiality",
    "post_title",
    "post_upvote_ratio",
    "post_score",
    "post_created_time",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to the full CSV")
    parser.add_argument("output_parquet", help="Path to write the parquet file")
    parser.add_argument("--n", type=int, default=5000, help="Number of rows to sample")
    parser.add_argument(
        "--mode",
        choices=["head", "random"],
        default="random",
        help="'head' = first N rows (fast, no full read). "
        "'random' = random sample across the whole file (slower, needs full read).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "head":
        # Fast path: only reads the first N rows, never loads the full file.
        df = pd.read_csv(args.input_csv, nrows=args.n)
    else:
        # Random sample across the whole file. Reads the full CSV once, so it's
        # slower on a big file — worth it since 'head' would only ever give you
        # the most recent comments in whatever order the CSV is sorted.
        df = pd.read_csv(args.input_csv)
        n = min(args.n, len(df))
        df = df.sample(n=n, random_state=args.seed).reset_index(drop=True)

    print(f"Loaded/sampled shape: {df.shape}")

    # Keep only columns that actually exist in this file
    cols_present = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"Note: these expected columns were not found and will be skipped: {missing}")

    df = df[cols_present]

    # Basic cleanup: drop rows with empty/near-empty text, common Reddit placeholders
    before = len(df)
    df["self_text"] = df["self_text"].astype(str)
    df = df[~df["self_text"].isin(["[deleted]", "[removed]", "nan", "NaN", ""])]
    df = df[df["self_text"].str.len() >= 10]
    print(f"Dropped {before - len(df)} rows with empty/placeholder/short text")

    # Parse timestamp for later time-series work
    if "created_time" in df.columns:
        df["created_time"] = pd.to_datetime(df["created_time"], errors="coerce")

    df = df.reset_index(drop=True)
    df.to_parquet(args.output_parquet, index=False)
    print(f"Saved {len(df)} rows x {len(df.columns)} cols -> {args.output_parquet}")
    print(f"Columns kept: {list(df.columns)}")


if __name__ == "__main__":
    main()
