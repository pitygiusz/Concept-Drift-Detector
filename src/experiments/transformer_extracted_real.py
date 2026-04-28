from pathlib import Path


from src.models.sampling_schemas import extract_from_all_partitions
from src.models.transformer import run_experiment, plot_results

DATA_DIR = Path("data/processed")

def main():
    # 1. Load and extract from all partitions
    sorted_stream3 = extract_from_all_partitions(DATA_DIR)

    print(f"Loaded samples: {len(sorted_stream3)}")

    if len(sorted_stream3) == 0:
        print("No data loaded. Check DATA_DIR or class balance.")
        return

    # 2. Run experiment
    dates, acc = run_experiment(sorted_stream3)

    # 3. Plot results
    header = "TF-IDF + MultinomialNB - Max 1 per Class per Day"
    plot_results(dates, acc, sorted_stream3)


if __name__ == "__main__":
    main()

