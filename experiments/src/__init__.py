"""
Main package exports for the concept drift detection project.
"""

from .plot_experiment import plot_results, plot_shadow_models

from .preprocess_real_data import (
    extract_max_n_per_class_per_day,
    undersample_stream,
    split_text_into_parts,
    extract_and_split_minority_per_day,
)

from .run_experiment import run_experiment

from .synthetic_stream_generator import (
    create_fresh_model,
    SyntheticPoliticalStream,
    LEFT_VOCAB,
    RIGHT_VOCAB,
    NEUTRAL_VOCAB,
)

__all__ = [
    "plot_results",
    "plot_shadow_models",
    "extract_max_n_per_class_per_day",
    "undersample_stream",
    "split_text_into_parts",
    "extract_and_split_minority_per_day",
    "run_experiment",
    "create_fresh_model",
    "SyntheticPoliticalStream",
    "LEFT_VOCAB",
    "RIGHT_VOCAB",
    "NEUTRAL_VOCAB",
]