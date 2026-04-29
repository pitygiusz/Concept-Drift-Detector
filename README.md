# Concept Drift Detection in Polarized News Streams

An online machine learning project investigating **concept drift in political language**, combining real-world news data, streaming ML, and unsupervised semantic analysis.


##  Overview

Political language evolves rapidly, especially around major events like elections, debates, or candidate changes. This evolution introduces **concept drift** — a shift in the relationship between text and its meaning over time.

This project explores:

-  Why **classification-based drift detection fails** in real-world political data  
-  How **unsupervised distributional methods succeed**  
-  The difference between **synthetic vs real-world drift behavior**



##  Key Insight

> **Drift detection is useless if your model learns nothing.**

Our experiments include:

- Classification models (**TF-IDF + NB**, **LLM + Logistic Regression**)  
  - collapse to **majority-class prediction**  on real-world data, **no drift detection possible**
  - perform well on synthetic data where signal is strong

- Distributional approach (**cosine distance between embeddings**)  
  - detects **real-world semantic shifts aligned with events**

Main Contributions include:
- Custom-made synthetic natural language data generator for controlled drift experiments
- Extending the `river.base.Transformer` class to combine onlne learning with transformers-based embeddings
- Cutom-made unsupervised drift detection pipeline using ADWIN, Page-Hinkley, and KSWIN on embedding distances



##  Methodology

###  Part I — Classification-Based Drift (Piotr Jurczyk)

#### Pipeline:
1. Synthetic data generation
2. TF-IDF + MultinomialNB
3. SentenceTransformer + Logistic Regression
4. ADWIN drift detection

#### Result:
- Works on synthetic data   
- **Fails completely on real data** - models default to majority class



###  Part II — Distributional Drift (Krzysztof Krawiec)

#### Pipeline:

1. Group articles (daily/weekly)
2. Compute embeddings (MiniLM)
3. Measure cosine distance between windows
4. Apply drift detectors: ADWIN, Page-Hinkley, KSWIN

#### Result:
- Detects **real-world semantic shifts** aligned with political events
- More robust to noise and sparsity than classification approach

##  Experiments (Current State)

At this stage of the project, most experiments are conducted and organized within the `notebooks/` directory.  
These notebooks serve as the primary interface for running analyses, visualizing results, and iterating on ideas.

While this approach enables fast experimentation and flexibility during development, the structure is currently **not fully modularized**.

 In future iterations, the project will be refactored to:
- move core experiment logic into reusable Python modules (`src/experiments/`)
- provide clean CLI entry points for running experiments
- standardize configuration and result logging
- separate exploratory analysis from reproducible pipelines

The long-term goal is to transition from a **research-oriented notebook workflow** to a **fully structured, production-ready experiment framework**.

##  Installation

```bash
git clone https://github.com/pitygiusz/Concept-Drift-Detector.git
cd Concept-Drift-Detector

python -m venv .venv
source .venv/bin/activate  # or Windows equivalent

pip install -r requirements.txt


