# =========================
# BASELINE (TF-IDF + NB)
# =========================
from .baseline import run_experiment as baseline_run
from .baseline import plot_results as baseline_plot

# =========================
# SYNTHETIC (ADWIN + shadow models)
# =========================
from .synthetic_baseline import (
    create_fresh_model,
    run_adaptive_experiment,
    plot_shadow_models,
)

# =========================
# TRANSFORMER MODEL
# =========================
from .transformer import (
    run_experiment as transformer_run,
    plot_results as transformer_plot,
)

# =========================
# DATA SAMPLING (you may move later)
# =========================
from .sampling_schemas import (
    process_all_partitions,
    extract_from_all_partitions,
)

__all__ = [
    # baseline
    "baseline_run",
    "baseline_plot",

    # synthetic
    "create_fresh_model",
    "run_adaptive_experiment",
    "plot_shadow_models",

    # transformer
    "transformer_run",
    "transformer_plot",

    # sampling
    "process_all_partitions",
    "extract_from_all_partitions",
]