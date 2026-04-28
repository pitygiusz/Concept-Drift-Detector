import json
from pathlib import Path

from src.data_acquisition.synthetic_stream import SyntheticPoliticalStream



def save_synthetic_stream(stream_type = "basic"):
    output_file = Path(f"data/synthetic/synthetic_stream_{stream_type}.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    generator = SyntheticPoliticalStream(
        stream_type= stream_type,
        seed=42
    )

    if stream_type == "basic":
        stream = generator.get_stream(
            n_samples=3000,
            drift_points=[1000, 2000],
            drift_ratio=0.6,
            min_len=20,
            max_len=50
        )
    else:
        stream = generator.get_stream(
            n_samples=3000,
            abrupt_drifts=[(800, 0.5)],
            gradual_drifts=[(1500, 2000, 0.9)],
            recurring_drifts=[2500],
            min_len=20,
            max_len=50
        )

    with output_file.open("w", encoding="utf-8") as f:
        for i, (text, label) in enumerate(stream):
            row = {
                "id": i,
                "timestamp": i,
                "text": text,
                "label": label,
                "source": "synthetic",
                "stream_type": stream_type
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved synthetic data to: {output_file}")


if __name__ == "__main__":
    save_synthetic_stream("basic")