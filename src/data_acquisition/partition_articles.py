import json
from datetime import datetime
from pathlib import Path
import shutil


INPUT_FILE = Path("data/raw/articles_raw.jsonl")
OUTPUT_DIR = Path("data/processed")


def get_partition_path(output_dir, seen_date):
    """
    Create path:
    data/processed/2024/10.jsonl
    """

    try:
        date = datetime.strptime(seen_date[:8], "%Y%m%d")
    except Exception:
        date = datetime.now()

    year_dir = output_dir / str(date.year)
    year_dir.mkdir(parents=True, exist_ok=True)

    return year_dir / f"{date.month:02d}.jsonl"


def save_jsonl_row(output_file, row):
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def partition_articles():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file does not exist: {INPUT_FILE}")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    saved_rows = 0
    skipped_invalid = 0

    print("=" * 70)
    print("Starting partitioning")
    print(f"Input file:  {INPUT_FILE}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print("=" * 70)

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            total_rows += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped_invalid += 1
                continue

            seen_date = row.get("seen_date")

            partition_file = get_partition_path(
                output_dir=OUTPUT_DIR,
                seen_date=seen_date
            )

            save_jsonl_row(partition_file, row)
            saved_rows += 1

    print("\n" + "=" * 70)
    print("Finished partitioning")
    print(f"Total rows read:     {total_rows}")
    print(f"Saved rows:          {saved_rows}")
    print(f"Invalid rows skipped:{skipped_invalid}")
    print(f"Output dir:          {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    partition_articles()