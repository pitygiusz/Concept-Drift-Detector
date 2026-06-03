# Concept Drift Detection in Polarized News Streams

An online machine learning project investigating **concept drift in political language**, combining real-world news data, streaming ML, and unsupervised semantic analysis.

## Abstract
Online political discourse constantly evolves, creating potential concept drift that degrades static NLP models. This paper develops and evaluates two complementary approaches for drift detection in news streams sourced from the GDELT database spanning the 2024 U.S. Presidential Election. 

In the first part (Classification-based drift), we compare various vectorization and online classifier strategies on synthetic and real-world data. We use the ADWIN drift detector to detect drift in the stream of errors produced by classifiers.

In the second part (Distributional-based drift), we apply an unsupervised distributional drift pipeline using daily and weekly article aggregates. Cosine distances are compared for consecutive windows in TF-IDF and embedding spaces, and the resulting distance streams are monitored with drift detectors. This avoids reliance on classification accuracy.

We show that given sufficient amount of data both approaches are able to detect shifts aligned with some major political events.


## Novel Methods

- Custom-made synthetic natural language data generator for controlled drift experiments (simulating abrupt, gradual, and recurring drifts).
- Design and implement three data balancing pipelines for real-world news data.
- Extending the `river.base.Transformer` class to combine online learning with pre-trained transformer-based embeddings (e.g., `politicalBiasBERT`, `all-MiniLM-L6-v2`).
- Custom-made unsupervised drift detection pipeline using ADWIN, Page-Hinkley, and KSWIN on univariate cosine distance streams.

## Methodology and Contributions

### Part I — Adaptive Streaming Classifiers (Piotr Jurczyk)

#### Pipeline:
1. Synthetic data generation & Real-world data preprocessing (GDELT news articles).
2. TF-IDF + MultinomialNB pipeline.
3. SentenceTransformer + SGD Logistic Regression pipeline.
4. ADWIN drift detection on the binary error stream.

#### Result:
- **Highly Successful on Real Data:** Overcame the majority-class baseline. Both models achieved strong rolling accuracy (80–95%).
- **Behavioral Discovery:** TF-IDF proved superior for active concept drift detection (capturing multiple rapid shifts during government transition), whereas LLMs proved better for stable, long-term classification.

### Part II — Unsupervised Distributional Drift (Krzysztof Krawiec)

#### Pipeline:
1. Group and aggregate articles into time windows (daily/weekly).
2. Compute contextual representations (TF-IDF and LLM embeddings) for aggregated documents.
3. Measure cosine distance between consecutive time windows.
4. Apply drift detectors (ADWIN, Page-Hinkley, KSWIN) directly to the distance stream.

#### Result:
- **Event Alignment:** Successfully detects real-world semantic shifts strictly aligned with major political events without requiring labeled data.
- **Detector Efficacy:** Page-Hinkley proved to be the most effective detector for capturing abrupt distributional shifts in political discourse.

## Project Structure

```
├── experiments/
│   ├── data/        #unprocessed and processed datasets 
│   ├── results/     #results of the distributional drift experiments
│   ├── src/         #source code for various functions and classes 
│   ├── 01_web_scraping.ipynb
│   ├── 02_synthetic_data_classification.ipynb
│   ├── 03_real_data_classification.ipynb
│   └── 04_sudden_drift_detection_enhanced.ipynb
├── README.md
└── requirements.txt
```