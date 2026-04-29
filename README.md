# Concept Drift Detector

A research-oriented Python project for detecting concept drift in political news streams.

The project focuses on monitoring how the language and classification behavior of polarized news sources changes over time, especially around major U.S. political events. It combines real-world news scraping, stream construction, online learning, drift detection methods, transformer embeddings, and synthetic drift experiments.

---

## Project Overview

Concept drift occurs when the statistical properties of a data stream change over time. In text classification, this may mean that the vocabulary, topics, tone, or relationship between text and labels changes after important real-world events.

This repository investigates concept drift in political news streams using:

- real article collection from selected news domains,
- temporal partitioning of news data,
- balanced and time-preserving stream sampling strategies,
- online text classification,
- drift detection with adaptive methods,
- transformer-based sentence embeddings,
- synthetic streams with controlled drift scenarios,
- visual analysis of drift alarms and event timelines.

The main use case is political language drift around events such as elections, candidate changes, inauguration, and other major campaign-related events.

---

## Repository Structure

```text
Concept-Drift-Detector/
│
├── data/
│   ├── raw/                 # Raw scraped articles
│   ├── processed/           # Articles partitioned by year/month
│   └── streams/             # Prepared data streams
│
├── notebooks/
│   ├── 01_1_web_scraping.ipynb
│   ├── 01_2_web_scraping.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_transformers.ipynb
│   ├── 04_data_generation.ipynb
│   ├── 05_sudden_drift_detection.ipynb
│   └── control_panel.ipynb
│
├── reports/                 # Report files and written analysis
├── results/                 # Generated plots and experiment outputs
│
├── src/
│   ├── data_acquisition/
│   │   ├── scrape_articles.py
│   │   ├── partition_articles.py
│   │   ├── run_pipeline.py
│   │   └── synthetic_stream.py
│   │
│   ├── experiments/
│   │   ├── baseline_balanced_real.py
│   │   ├── baseline_extracted_real.py
│   │   ├── transformer_extracted_real.py
│   │   ├── basic_synthetic.py
│   │   └── advanced_synthetic.py
│   │
│   └── models/
│       ├── baseline.py
│       ├── transformer.py
│       ├── sampling_schemas.py
│       └── synthetic_baseline.py
│
├── requirements.txt
├── LICENSE
└── README.md
