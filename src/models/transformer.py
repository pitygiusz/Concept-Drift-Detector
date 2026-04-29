from datetime import datetime
from collections import deque, Counter

import matplotlib.pyplot as plt
from river import base, compose, preprocessing, linear_model, optim, utils, metrics
from sentence_transformers import SentenceTransformer


def parse_timestamp(timestamp):
    timestamp = str(timestamp).replace("T", "")

    return datetime.strptime(timestamp[:14], "%Y%m%d%H%M%S")


class StreamSentenceTransformer(base.Transformer):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading SentenceTransformer model '{model_name}'...")
        self.encoder = SentenceTransformer(model_name)

    def transform_one(self, x):
        vector = self.encoder.encode(x)
        return {f"emb_{i}": float(value) for i, value in enumerate(vector)}


def run_experiment(data_stream):
    model = compose.Pipeline(
        ("vectorizer", StreamSentenceTransformer("all-MiniLM-L6-v2")),
        ("scaler", preprocessing.StandardScaler()),
        ("classifier", linear_model.LogisticRegression(
            optimizer=optim.SGD(0.05)
        ))
    )

    metric = utils.Rolling(metrics.Accuracy(), window_size=50)

    dates = []
    accuracies = []

    stream_list = list(data_stream)

    if len(stream_list) == 0:
        print("Empty stream.")
        return [], []

    latest_seen_date = parse_timestamp(stream_list[0]["timestamp"])

    for i, record in enumerate(stream_list):
        text = record["text"]
        label = record["label"]

        current_date = parse_timestamp(record["timestamp"])

        if current_date > latest_seen_date:
            latest_seen_date = current_date

        y_pred = model.predict_one(text)

        if y_pred is not None:
            metric.update(label, y_pred)
            dates.append(latest_seen_date)
            accuracies.append(metric.get())

        model.learn_one(text, label)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} articles out of {len(stream_list)}.")

    return dates, accuracies


def calculate_majority_baseline(sorted_stream, window_size=50):
    window_labels = deque(maxlen=window_size)

    dates_baseline = []
    majority_acc = []

    stream_list = list(sorted_stream)

    if len(stream_list) == 0:
        return [], []

    latest_seen_date = parse_timestamp(stream_list[0]["timestamp"])

    for record in stream_list:
        current_date = parse_timestamp(record["timestamp"])

        if current_date > latest_seen_date:
            latest_seen_date = current_date

        label = record["label"]
        window_labels.append(label)

        most_common_count = Counter(window_labels).most_common(1)[0][1]
        acc = most_common_count / len(window_labels)

        dates_baseline.append(latest_seen_date)
        majority_acc.append(acc)

    return dates_baseline, majority_acc


def plot_results(model_dates, model_acc, sorted_stream):
    baseline_dates, baseline_acc = calculate_majority_baseline(sorted_stream)

    plt.figure(figsize=(14, 7))

    plt.plot(
        baseline_dates,
        baseline_acc,
        color="red",
        linewidth=3,
        linestyle="--",
        label="Majority Class"
    )

    plt.plot(
        model_dates,
        model_acc,
        color="green",
        linewidth=1.5,
        alpha=0.8,
        label="SentenceTransformer + Logistic Regression"
    )

    plt.title(
        "Elections Impact on Press in USA",
        fontsize=16,
        fontweight="bold"
    )

    plt.axvline(
        x=datetime(2024, 11, 5),
        color="purple",
        linestyle="-",
        linewidth=2,
        label="2024 US Election"
    )

    plt.ylim(0.0, 1.05)
    plt.xlabel("Date")
    plt.ylabel("Rolling Accuracy")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()