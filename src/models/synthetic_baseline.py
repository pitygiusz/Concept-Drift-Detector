import matplotlib.pyplot as plt
from river import compose, feature_extraction, naive_bayes, metrics, drift, utils


def create_fresh_model():
    return compose.Pipeline(
        ("tfidf", feature_extraction.TFIDF()),
        ("nb", naive_bayes.MultinomialNB())
    )


def run_adaptive_experiment(stream, train_strategy="all", reset_on_drift=True):
    models_info = [{
        "model": create_fresh_model(),
        "metric": utils.Rolling(metrics.Accuracy(), window_size=100),
        "accuracies": [],
        "start_step": 0
    }]

    detector = drift.ADWIN(delta=0.02)

    steps = []
    drifts_detected = []

    for i, (text, y_true) in enumerate(stream):
        steps.append(i)
        active_model_idx = len(models_info) - 1

        for idx, info in enumerate(models_info):
            y_pred = info["model"].predict_one(text)

            if y_pred is not None:
                info["metric"].update(y_true, y_pred)

                if idx == active_model_idx:
                    error = 0 if y_pred == y_true else 1
                    detector.update(error)

            info["accuracies"].append(info["metric"].get())

            if train_strategy == "all":
                info["model"].learn_one(text, y_true)

            elif train_strategy == "active_only":
                if idx == active_model_idx:
                    info["model"].learn_one(text, y_true)

            else:
                raise ValueError(f"Unknown train_strategy: {train_strategy}")

        if detector.drift_detected:
            drifts_detected.append(i)
            print(f"Step {i}: Concept drift detected on Model {active_model_idx + 1}")

            if reset_on_drift:
                new_info = {
                    "model": create_fresh_model(),
                    "metric": utils.Rolling(metrics.Accuracy(), window_size=100),
                    "accuracies": [None] * (i + 1),
                    "start_step": i + 1
                }

                models_info.append(new_info)
                detector = drift.ADWIN(delta=0.02)

    return steps, models_info, drifts_detected


def plot_shadow_models(steps, models_info, drifts_detected, drift_points, title):
    plt.figure(figsize=(14, 7))

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c",
        "#d62728", "#9467bd", "#8c564b"
    ]

    for idx, info in enumerate(models_info):
        color = colors[idx % len(colors)]
        label = f"Model {idx + 1} active from step {info['start_step']}"

        plt.plot(
            steps,
            info["accuracies"],
            label=label,
            color=color,
            linewidth=2.5
        )

    for i, drift_step in enumerate(drifts_detected):
        label = "ADWIN alert and reset" if i == 0 else None

        plt.axvline(
            x=drift_step,
            color="red",
            linestyle="--",
            alpha=0.8,
            label=label
        )

    for i, drift_point in enumerate(drift_points):
        label = "Actual drift injection" if i == 0 else None

        plt.axvline(
            x=drift_point,
            color="purple",
            linestyle="-",
            linewidth=2,
            label=label
        )

    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel("Number of processed articles")
    plt.ylabel("Rolling Accuracy")
    plt.ylim(0.0, 1.05)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.show()