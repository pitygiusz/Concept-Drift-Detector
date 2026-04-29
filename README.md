# Concept Drift Detection in Polarized News Streams

An online machine learning project investigating **concept drift in political language**, combining real-world news data, streaming ML, and unsupervised semantic analysis.

---

##  Overview

Political language evolves rapidly, especially around major events like elections, debates, or candidate changes. This evolution introduces **concept drift** — a shift in the relationship between text and its meaning over time.

This project explores:

-  Why **classification-based drift detection fails** in real-world political data  
-  How **unsupervised distributional methods succeed**  
-  The difference between **synthetic vs real-world drift behavior**

---

##  Key Insight (Main Contribution)

> **Drift detection is useless if your model learns nothing.**

From experiments:

- Classification models (**TF-IDF + NB**, **LLM + Logistic Regression**)  
  → collapse to **majority-class prediction**  
  → **no drift detection possible**

- Distributional approach (**cosine distance between embeddings**)  
  → detects **real-world semantic shifts aligned with events**

 This is the core conclusion of the project 

---

##  Methodology

###  Part I — Classification-Based Drift

Pipeline:

Tested:
- TF-IDF + MultinomialNB
- SentenceTransformer + Logistic Regression
- ADWIN drift detection

### Result:
- Works on synthetic data   
- **Fails completely on real data **

Reason:
- Political text is **too noisy and sparse**
- Models default to majority class

---

###  Part II — Distributional Drift

Pipeline:

Steps:
1. Group articles (daily/weekly)
2. Compute embeddings (MiniLM)
3. Measure cosine distance between windows
4. Apply drift detectors:
   - ADWIN
   - Page-Hinkley
   - KSWIN

### Result:
- Detects meaningful shifts
- Aligns with political events
- Works without labels

---

##  Experimental Results

###  Synthetic Data

- ADWIN detects abrupt drift reliably  
- Gradual drift → harder to detect  
- Recurring drift → detected strongly  

 Validates correctness of implementation

---

###  Real-World Data (GDELT)

#### Classification Approach

- Accuracy ≈ majority baseline
- No correlation with:
  - US Election (Nov 2024)
  - Inauguration (Jan 2025)

 Model learns nothing → drift detection impossible

---

####  Distributional Drift

##### Mixed stream:
- Detects events like:
  - election period
  - conventions

##### Ideology-specific streams:

 Left-leaning:
- Biden withdrawal
- Harris nomination
- Democratic Convention

 Right-leaning:
- Trump conviction
- Assassination attempt
- Republican Convention

 **Drift is ideology-dependent** 

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


