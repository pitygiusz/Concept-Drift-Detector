import requests
import trafilatura
import json
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pathlib import Path

# A script for gettin URL from GDELT and scraping using trafilatura

# ============================================================
# CONFIG
# ============================================================

SOURCES = {
    # LEFT
    "thenation.com": 0,
    # "motherjones.com": 0,
    # "vox.com": 0,

    # RIGHT
    # "breitbart.com": 1,
    # "foxnews.com": 1,
    # "dailycaller.com": 1,
}

KEYWORDS = "Trump OR Biden OR President"

START_DATE = datetime(2025, 12, 27)
END_DATE = datetime(2026, 4, 27)

OUTPUT_FILE = Path("data/raw/articles_raw.jsonl")

WINDOW_SIZE_MONTHS = 1
MAX_RECORDS_PER_REQUEST = 100

REQUEST_DELAY = 3
SCRAPE_DELAY = 1

MAX_RETRIES = 10


# ============================================================
# GDELT API
# ============================================================

def get_article_urls_from_gdelt(domain, query, start_date, end_date, max_retries=MAX_RETRIES):
    """
    Download article URLs from GDELT API for a selected domain and date range.
    """

    url = "https://api.gdeltproject.org/api/v2/doc/doc"

    start_str = start_date.strftime("%Y%m%d%H%M%S")
    end_str = end_date.strftime("%Y%m%d%H%M%S")

    full_query = f"({query}) domain:{domain}"

    params = {
        "query": full_query,
        "mode": "artlist",
        "maxrecords": MAX_RECORDS_PER_REQUEST,
        "sort": "DateDesc",
        "format": "json",
        "startdatetime": start_str,
        "enddatetime": end_str,
    }

    for attempt in range(1, max_retries + 1):
        try:
            time.sleep(REQUEST_DELAY)

            response = requests.get(url, params=params, timeout=30)

            if response.status_code == 429:
                wait_time = 5 * attempt
                print(f"   [API] Rate limit. Waiting {wait_time}s. Attempt {attempt}/{max_retries}")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])

            if not articles:
                print(f"   [API] 0 articles found. Attempt {attempt}/{max_retries}")
                time.sleep(3)
                continue

            results = []

            for article in articles:
                article_url = article.get("url")
                seen_date = article.get("seendate")

                if article_url:
                    results.append((article_url, seen_date))

            return results

        except requests.exceptions.Timeout:
            print(f"   [API] Timeout. Attempt {attempt}/{max_retries}")
            time.sleep(5)

        except requests.exceptions.ConnectionError:
            print(f"   [API] Connection error. Attempt {attempt}/{max_retries}")
            time.sleep(5)

        except requests.exceptions.HTTPError as e:
            print(f"   [API] HTTP error: {e}. Attempt {attempt}/{max_retries}")
            time.sleep(5)

        except json.JSONDecodeError:
            print(f"   [API] Invalid JSON response. Attempt {attempt}/{max_retries}")
            time.sleep(5)

    print(f"   [API] Failed after {max_retries} attempts.")
    return []


# ============================================================
# ARTICLE SCRAPER
# ============================================================

def scrape_article_text(url):
    """
    Download and extract main article text using Trafilatura.
    """

    try:
        downloaded = trafilatura.fetch_url(url)

        if not downloaded:
            return None

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            favor_precision=True
        )

        return text

    except Exception as e:
        print(f"   [SCRAPER] Error scraping {url}: {e}")
        return None
    
# ============================================================
# HELPERS
# ============================================================

def save_jsonl_row(output_file, row):
    """
    Append one record to JSONL file.
    """
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing_urls(output_file):
    """
    Load URLs already saved in JSONL file to avoid duplicates.
    """

    seen_urls = set()

    if not output_file.exists():
        return seen_urls

    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                if "url" in row:
                    seen_urls.add(row["url"])
            except json.JSONDecodeError:
                continue

    return seen_urls


# ============================================================
# MAIN FUNCTION
# ============================================================

def scrape_data():
    seen_urls = load_existing_urls(OUTPUT_FILE)

    total_found = 0
    total_saved = 0
    total_duplicates = 0
    total_failed_scrapes = 0

    print("=" * 70)
    print("Starting GDELT article collection")
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Already collected URLs: {len(seen_urls)}")
    print("=" * 70)

    for domain, label in SOURCES.items():
        print(f"\nSource: {domain}, label: {label}")

        current_date = START_DATE

        while current_date < END_DATE:
            next_date = current_date + relativedelta(months=WINDOW_SIZE_MONTHS)

            if next_date > END_DATE:
                next_date = END_DATE

            print(f"\nCollecting articles from {current_date.date()} to {next_date.date()}")

            articles = get_article_urls_from_gdelt(
                domain=domain,
                query=KEYWORDS,
                start_date=current_date,
                end_date=next_date
            )

            print(f"   Found URLs: {len(articles)}")
            total_found += len(articles)

            for url, seen_date in articles:
                if url in seen_urls:
                    total_duplicates += 1
                    continue

                time.sleep(SCRAPE_DELAY)

                text = scrape_article_text(url)

                if text is None:
                    total_failed_scrapes += 1
                    continue

                row = {
                    "url": url,
                    "domain": domain,
                    "label": label,
                    "seen_date": seen_date,
                    "text": text,
                    "text_length": len(text),
                    "collected_at": datetime.now().isoformat(timespec="seconds")
                }

                save_jsonl_row(OUTPUT_FILE, row)

                seen_urls.add(url)
                total_saved += 1

                print(f"   Saved article #{total_saved}: {url}")

            current_date = next_date

    print("\n" + "=" * 70)
    print("Finished")
    print(f"Total URLs found:       {total_found}")
    print(f"Saved articles:         {total_saved}")
    print(f"Duplicates skipped:     {total_duplicates}")
    print(f"Failed scrapes:         {total_failed_scrapes}")
    print(f"Output file:            {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    scrape_data()