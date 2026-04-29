from src.data_acquisition.scrape_articles import scrape_data
from src.data_acquisition.partition_articles import partition_articles
from src.models.sampling_schemas import (
    process_all_partitions,
    extract_from_all_partitions,
    save_jsonl
)



def main(run_scraping=True, run_partitioning=True, run_sampling=True):
    if run_scraping:
        print("\n=== STEP 1: SCRAPING ===")
        scrape_data()

    if run_partitioning:
        print("\n=== STEP 2: PARTITIONING ===")
        partition_articles()

    if run_sampling:
        print("\n=== STEP 3: BALANCED STREAM ===")
        balanced = process_all_partitions("data/processed")
        save_jsonl(balanced, "data/streams/balanced_stream.jsonl")

        print("\n=== STEP 4: DAILY STREAM ===")
        daily = extract_from_all_partitions("data/processed")
        save_jsonl(daily, "data/streams/one_per_class_per_day_stream.jsonl")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()