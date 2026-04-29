from src.data_acquisition.synthetic_stream import SyntheticPoliticalStream
from src.models.synthetic_baseline import run_adaptive_experiment, plot_shadow_models


def main():
    drift_points = [800, 1600, 2400]
    total_samples = 3200

    generator = SyntheticPoliticalStream(stream_type="basic")

    stream = generator.get_stream(
        n_samples=total_samples,
        drift_points=drift_points,
        drift_ratio=0.5,
        min_len=20,
        max_len=50
    )

    steps, models_info, alerts = run_adaptive_experiment(
        stream=stream,
        train_strategy="all",
        reset_on_drift=True
    )

    plot_shadow_models(
        steps=steps,
        models_info=models_info,
        drifts_detected=alerts,
        drift_points=drift_points,
        title="Basic Synthetic Stream: TF-IDF + MultinomialNB + ADWIN"
    )


if __name__ == "__main__":
    main()