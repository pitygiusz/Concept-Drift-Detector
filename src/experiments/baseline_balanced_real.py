from pathlib import Path

from src.models.sampling_schemas import process_all_partitions  # sampling schema 1
from src.models.baseline import run_experiment, plot_results


DATA_DIR = Path("data/processed")

def main():
    # 1. Load and balance real partitioned stream
    sorted_stream1 = process_all_partitions(DATA_DIR)

    print(f"Loaded samples: {len(sorted_stream1)}")

    if len(sorted_stream1) == 0:
        print("No data loaded. Check DATA_DIR or class balance.")
        return

    # 2. Run experiment
    steps, acc, drifts = run_experiment(sorted_stream1)

    # 3. Plot results
    header = "TF-IDF + MultinomialNB - Undersampling"
    plot_results(steps, acc, drifts, header)


if __name__ == "__main__":
    main()
