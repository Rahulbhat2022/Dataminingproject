"""
Combined Reddit comment cleaning + CSV -> Parquet conversion.

Merges the two-stage cleaning rules from DM1_01_combined_cleaning.ipynb with
the CSV->Parquet conversion approach from csv_to_parquet.py, and runs the
full pipeline over the entire reddit_opinion_climate_change.csv (no
sampling, no column subsetting).

The raw CSV contains a small number of records with an embedded-quote
pattern (e.g. `""word`) that breaks pandas' C parser (it reports
"EOF inside string"). Python's built-in `csv` module handles this file
correctly (verified: it reads exactly 1,903,836 data rows with a
consistent 24 columns each), so this script streams the file with `csv`
instead of `pandas.read_csv`. Streaming in chunks also keeps memory bounded
on the full ~1.9M-row / 2.5GB file, and gives a natural place to report
progress.

Usage:
    python3 clean_and_convert.py reddit_opinion_climate_change.csv \
        reddit_opinion_climate_change_cleaned_final.parquet \
        --audit-path reddit_cleaning_removed_audit.parquet \
        --report-path reddit_cleaning_report_final.csv
"""

import argparse
import csv
import html
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

ORIGINAL_COLUMNS = [
    "comment_id", "score", "self_text", "subreddit", "created_time", "post_id",
    "author_name", "controversiality", "ups", "downs", "user_is_verified",
    "user_account_created_time", "user_awardee_karma", "user_awarder_karma",
    "user_link_karma", "user_comment_karma", "user_total_karma", "post_score",
    "post_self_text", "post_title", "post_upvote_ratio", "post_thumbs_ups",
    "post_total_awards_received", "post_created_time",
]

KARMA_COLUMNS = [
    "user_awardee_karma", "user_awarder_karma", "user_link_karma",
    "user_comment_karma", "user_total_karma",
]

NUMERIC_COLUMNS = [
    "score", "controversiality", "ups", "downs", "post_score",
    "post_upvote_ratio", "post_thumbs_ups", "post_total_awards_received",
] + KARMA_COLUMNS

DATETIME_COLUMNS = ["created_time", "post_created_time", "user_account_created_time"]

FINAL_COLUMNS = ORIGINAL_COLUMNS + [
    "clean_text", "comment_char_count", "comment_word_count",
    "has_post_self_text", "user_metadata_available",
]

AUDIT_COLUMNS = ORIGINAL_COLUMNS + ["clean_text", "removal_stage", "removal_reason"]

MOD_AUTHOR_RE = re.compile(r"automoderator|modteam|read-the-rules|ukbot|revddit")
DELETION_PLACEHOLDERS = {"[deleted]", "[removed]", "[removed by reddit]", "deleted", "removed"}

INITIAL_REMOVAL_NOTICE_RE = re.compile(
    r"^(?:your |this )?(?:comment|post)\s+(?:has been|was|is)\s+removed\b"
    r"|^comment\s+removed\s+by\s+(?:moderator|reddit)\b"
    r"|^removed\s+for\s+(?:breaking|violating)\b"
)
BOT_FOOTER_RE = re.compile(r"i am a bot, and this action was performed automatically")
BOT_FOOTER_CONTEXT_RE = re.compile(r"remov|rule|moderator")

EXPLICIT_COMMENT_PHRASE_RE = re.compile(
    r"\b(?:your|this|the)\s+comments?(?:\s+here\s+and\s+below)?\s+(?:has|have)\s+been\s+removed\b"
)
NOTICE_STRUCTURE_START_RE = re.compile(
    r"^\s*(?:sorry[,!]?\s*)?(?:u/[^\s,]+.*?\s+)?(?:your|this|the)\s+comments?"
)
BREAKING_RULE_RE = re.compile(r"\bbreaking\s+rule\s*[0-9a-z]?\b")
APPEAL_RE = re.compile(r"\bappeal(?:s|ing)?\b.{0,100}\b(?:moderator|removal|removed)\b")
POST_REMOVAL_RE = re.compile(
    r"^\s*(?:sorry[,!]?\s*)?(?:u/[^\s,]+.*?\s+)?your\s+post\s+has\s+been\s+removed\s+for\s+breaking\s+rule"
)
MOD_TEMPLATE_TEXT_RE = re.compile(r"removed|breaking\s+rule|reviewing\s+removed\s+content")

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)]+\)")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_RE = re.compile(r"\s+")


def normalize_comment_text(value):
    value = html.unescape(str(value))
    value = MARKDOWN_LINK_RE.sub(r"\1", value)
    value = URL_RE.sub(" ", value)
    value = WHITESPACE_RE.sub(" ", value)
    return value.strip()


def build_raw_frame(rows):
    df = pd.DataFrame(rows, columns=ORIGINAL_COLUMNS, dtype="object")
    df = df.mask(df == "")
    return df


def coerce_final_dtypes(df):
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    for col in DATETIME_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["user_is_verified"] = df["user_is_verified"].map(
        {"True": True, "False": False}
    ).astype("boolean")
    for col in ["comment_id", "self_text", "subreddit", "post_id", "author_name",
                "post_self_text", "post_title", "clean_text"]:
        df[col] = df[col].astype("string")
    df["has_post_self_text"] = df["has_post_self_text"].astype("boolean")
    df["user_metadata_available"] = df["user_metadata_available"].astype("boolean")
    df["comment_char_count"] = df["comment_char_count"].astype("Int64")
    df["comment_word_count"] = df["comment_word_count"].astype("Int64")
    return df


def build_final_schema():
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in FINAL_COLUMNS})
    df = coerce_final_dtypes(df)
    return pa.Schema.from_pandas(df[FINAL_COLUMNS])


def clean_chunk(rows, seen_ids, rule_counts, subreddit_before, subreddit_removed, audit_rows):
    df = build_raw_frame(rows)

    for sr in df["subreddit"].dropna():
        subreddit_before[sr] += 1

    raw_text = df["self_text"].astype("string")
    stripped_text = raw_text.str.strip()
    lower_text = stripped_text.str.lower()

    dup_mask = []
    for cid in df["comment_id"]:
        dup_mask.append(cid in seen_ids)
        seen_ids.add(cid)

    stage1_masks = {
        "missing_or_blank_comment_text": raw_text.isna() | stripped_text.eq(""),
        "deletion_placeholder": lower_text.isin(DELETION_PLACEHOLDERS).fillna(False),
        "initial_removal_notice": lower_text.str.contains(INITIAL_REMOVAL_NOTICE_RE, na=False),
        "automatic_removal_footer": (
            lower_text.str.contains(BOT_FOOTER_RE, na=False)
            & lower_text.str.contains(BOT_FOOTER_CONTEXT_RE, na=False)
        ),
        "duplicate_comment_id": pd.Series(dup_mask, index=df.index),
    }

    stage1_remove = pd.Series(False, index=df.index)
    stage1_reason = pd.Series("", index=df.index, dtype="object")
    for reason, mask in stage1_masks.items():
        new_matches = mask & ~stage1_remove
        stage1_reason.loc[new_matches] = reason
        stage1_remove |= mask
        rule_counts[("stage_1_raw_text", reason)] += int(new_matches.sum())

    working = df.loc[~stage1_remove].copy()
    working["clean_text"] = working["self_text"].map(normalize_comment_text)

    removed1 = df.loc[stage1_remove].copy()
    if len(removed1):
        removed1["clean_text"] = removed1["self_text"].fillna("").map(normalize_comment_text)
        removed1["removal_stage"] = "stage_1_raw_text"
        removed1["removal_reason"] = stage1_reason.loc[stage1_remove].values
        audit_rows.append(removed1[AUDIT_COLUMNS])

    empty_after_normalization = working["clean_text"].eq("")

    text = working["clean_text"].fillna("").astype(str).str.strip()
    text_lower = text.str.lower()
    author_lower = working["author_name"].fillna("").astype(str).str.lower()

    explicit_comment_phrase = text_lower.str.contains(EXPLICIT_COMMENT_PHRASE_RE, na=False)
    notice_structure = (
        text_lower.str.match(NOTICE_STRUCTURE_START_RE, na=False)
        | text_lower.str.contains(BREAKING_RULE_RE, na=False)
        | text_lower.str.contains(APPEAL_RE, na=False)
    )
    moderation_author = author_lower.str.contains(MOD_AUTHOR_RE, na=False)

    stage2_masks = {
        "empty_after_normalization": empty_after_normalization,
        "residual_deletion_placeholder": (
            text_lower.isin(["[deleted]", "[removed]", "deleted", "removed"])
            | text.isin(["None", "NaN", "<NA>"])
        ),
        "explicit_comment_removal_notice": explicit_comment_phrase & notice_structure,
        "explicit_post_removal_notice": text_lower.str.contains(POST_REMOVAL_RE, na=False),
        "recognized_moderation_template": moderation_author & text_lower.str.contains(
            MOD_TEMPLATE_TEXT_RE, na=False
        ),
    }

    stage2_remove = pd.Series(False, index=working.index)
    stage2_reason = pd.Series("", index=working.index, dtype="object")
    for reason, mask in stage2_masks.items():
        new_matches = mask & ~stage2_remove
        stage2_reason.loc[new_matches] = reason
        stage2_remove |= mask
        rule_counts[("stage_2_normalized_text", reason)] += int(new_matches.sum())

    removed2 = working.loc[stage2_remove].copy()
    if len(removed2):
        removed2["removal_stage"] = "stage_2_normalized_text"
        removed2["removal_reason"] = stage2_reason.loc[stage2_remove].values
        audit_rows.append(removed2[AUDIT_COLUMNS])

    for sr in removed1["subreddit"].dropna() if len(removed1) else []:
        subreddit_removed[sr] += 1
    for sr in removed2["subreddit"].dropna() if len(removed2) else []:
        subreddit_removed[sr] += 1

    df_clean = working.loc[~stage2_remove].copy()
    df_clean["comment_char_count"] = df_clean["clean_text"].str.len()
    df_clean["comment_word_count"] = df_clean["clean_text"].str.split().str.len()
    df_clean["has_post_self_text"] = df_clean["post_self_text"].notna()
    df_clean["user_metadata_available"] = df_clean[KARMA_COLUMNS].notna().all(axis=1)

    df_clean = coerce_final_dtypes(df_clean)
    return df_clean[FINAL_COLUMNS]


def count_data_rows(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        return sum(1 for _ in reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_parquet")
    parser.add_argument("--audit-path", default=None,
                         help="Optional path for the removal audit (parquet or .csv.gz)")
    parser.add_argument("--report-path", default=None,
                         help="Optional path for the per-rule cleaning report (csv)")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--skip-count", action="store_true",
                         help="Skip the upfront row-count pass (progress bar becomes indeterminate)")
    args = parser.parse_args()

    input_path = Path(args.input_csv)

    total_rows = None
    if not args.skip_count:
        print("Counting rows for progress bar...", file=sys.stderr)
        total_rows = count_data_rows(input_path)
        print(f"Total data rows: {total_rows:,}", file=sys.stderr)

    schema = build_final_schema()
    writer = pq.ParquetWriter(args.output_parquet, schema)

    seen_ids = set()
    rule_counts = Counter()
    subreddit_before = Counter()
    subreddit_removed = Counter()
    audit_rows = []

    total_in = 0
    total_out = 0

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ORIGINAL_COLUMNS, f"Unexpected header: {header}"

        pbar = tqdm(total=total_rows, unit="rows", desc="Cleaning + writing parquet")
        buffer = []
        for row in reader:
            buffer.append(row)
            if len(buffer) >= args.chunksize:
                df_clean = clean_chunk(
                    buffer, seen_ids, rule_counts, subreddit_before, subreddit_removed, audit_rows
                )
                writer.write_table(pa.Table.from_pandas(df_clean, schema=schema, preserve_index=False))
                total_in += len(buffer)
                total_out += len(df_clean)
                pbar.update(len(buffer))
                buffer = []
        if buffer:
            df_clean = clean_chunk(
                buffer, seen_ids, rule_counts, subreddit_before, subreddit_removed, audit_rows
            )
            writer.write_table(pa.Table.from_pandas(df_clean, schema=schema, preserve_index=False))
            total_in += len(buffer)
            total_out += len(df_clean)
            pbar.update(len(buffer))
        pbar.close()

    writer.close()

    print(f"\nOriginal rows: {total_in:,}")
    print(f"Final rows:    {total_out:,}")
    print(f"Removed rows:  {total_in - total_out:,}")
    print(f"Retention:     {total_out / total_in:.4%}")
    print(f"Saved cleaned data -> {args.output_parquet}")

    if args.audit_path and audit_rows:
        audit_df = pd.concat(audit_rows, ignore_index=True)
        if str(args.audit_path).endswith(".parquet"):
            audit_df.to_parquet(args.audit_path, index=False)
        else:
            audit_df.to_csv(args.audit_path, index=False, encoding="utf-8",
                             compression={"method": "gzip", "compresslevel": 1})
        print(f"Saved removal audit ({len(audit_df):,} rows) -> {args.audit_path}")

    if args.report_path:
        rule_report = pd.DataFrame(
            [{"stage": stage, "rule": rule, "exclusive_removed": count}
             for (stage, rule), count in rule_counts.items()]
        )
        rule_report.to_csv(args.report_path, index=False, encoding="utf-8-sig")
        print(f"Saved cleaning report -> {args.report_path}")


if __name__ == "__main__":
    main()
