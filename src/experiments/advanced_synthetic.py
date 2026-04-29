from src.data_acquisition.synthetic_stream import SyntheticPoliticalStream
from src.models.synthetic_baseline import run_adaptive_experiment, plot_shadow_models


def main():
    total_samples = 4000

    abrupt_drifts = [(800, 0.5)]
    gradual_drifts = [(1500, 2000, 0.9)]
    recurring_drifts = [3000]

    plot_drift_markers = [800, 1500, 2000, 3000]

    generator = SyntheticPoliticalStream(stream_type="advanced")

    stream = generator.get_stream(
        n_samples=total_samples,
        abrupt_drifts=abrupt_drifts,
        gradual_drifts=gradual_drifts,
        recurring_drifts=recurring_drifts,
        min_len=75,
        max_len=100
    )

    steps, models_info, alerts = run_adaptive_experiment(
        stream=stream,
        train_strategy="active_only",
        reset_on_drift=True
    )

    plot_shadow_models(
        steps=steps,
        models_info=models_info,
        drifts_detected=alerts,
        drift_points=plot_drift_markers,
        title="Advanced Synthetic Stream: Abrupt, Gradual and Recurring Drift"
    )


if __name__ == "__main__":
    main()