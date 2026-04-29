import json
import random
from pathlib import Path
from collections import defaultdict


RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def process_and_balance_jsonl(file_path):
    """
    Schema 1:
    Under-sample the majority class inside one JSONL file,
    then sort chronologically by seen_date.
    """
    parsed_data = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)

            parsed_data.append({
                "timestamp": obj["seen_date"],
                "label": obj["label"],
                "text": obj.get("text", "")
            })

    class_0 = [item for item in parsed_data if item["label"] == 0]
    class_1 = [item for item in parsed_data if item["label"] == 1]

    min_count = min(len(class_0), len(class_1))

    if min_count == 0:
        return []

    balanced_data = (
        random.sample(class_0, min_count)
        + random.sample(class_1, min_count)
    )

    balanced_data.sort(key=lambda x: x["timestamp"])

    return balanced_data


def process_all_partitions(data_dir):
    """
    Apply Schema 1 to all partitioned JSONL files.
    Expected structure:
        data/processed/2024/10.jsonl
        data/processed/2024/11.jsonl
        ...
    """
    all_data = []

    for year_dir in sorted(Path(data_dir).iterdir()):
        if not year_dir.is_dir():
            continue

        for file_path in sorted(year_dir.glob("*.jsonl")):
            print(f"Processing {file_path}")
            balanced = process_and_balance_jsonl(file_path)
            all_data.extend(balanced)

    all_data.sort(key=lambda x: x["timestamp"])

    return all_data


def extract_max_one_per_class_per_day(input_data):
    """
    Schema 2:
    Extract max one article per day per class from opened JSONL input.
    This may produce an unbalanced dataset, but preserves temporal distribution better.
    """
    grouped_data = defaultdict(lambda: {0: [], 1: []})

    for line in input_data:
        if not line.strip():
            continue

        obj = json.loads(line)

        timestamp = obj["seen_date"]
        label = obj["label"]

        date_only = timestamp[:8]

        extracted_obj = {
            "timestamp": timestamp,
            "label": label,
            "text": obj.get("text", "")
        }

        grouped_data[date_only][label].append(extracted_obj)

    selected_data = []

    for date, classes in grouped_data.items():
        class_0_items = classes[0]
        class_1_items = classes[1]

        if class_0_items:
            selected_data.append(random.choice(class_0_items))

        if class_1_items:
            selected_data.append(random.choice(class_1_items))

    selected_data.sort(key=lambda x: x["timestamp"])

    return selected_data


def extract_from_all_partitions(data_dir):
    """
    Apply Schema 2 to all partitioned JSONL files.
    """
    all_data = []

    for year_dir in sorted(Path(data_dir).iterdir()):
        if not year_dir.is_dir():
            continue

        for file_path in sorted(year_dir.glob("*.jsonl")):
            print(f"Processing {file_path}")

            with open(file_path, "r", encoding="utf-8") as f:
                selected = extract_max_one_per_class_per_day(f)

            all_data.extend(selected)

    all_data.sort(key=lambda x: x["timestamp"])

    return all_data


def save_jsonl(data, output_file):
    """
    Save list of dictionaries to JSONL.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    DATA_DIR = "data/processed"

    balanced_data = process_all_partitions(DATA_DIR)
    save_jsonl(balanced_data, "data/streams/balanced_stream.jsonl")

    daily_data = extract_from_all_partitions(DATA_DIR)
    save_jsonl(daily_data, "data/streams/one_per_class_per_day_stream.jsonl")

