# =========================
# RAW DATA PIPELINE
# =========================
from .run_pipeline import main as run_data_pipeline


# =========================
# SYNTHETIC DATA
# =========================
from .make_synthetic_data import save_synthetic_stream

# =========================
# OPTIONAL (fine-grained steps)
# =========================
from .scrape_articles import scrape_data
from .partition_articles import partition_articles

__all__ = [
    # main pipeline
    "run_data_pipeline",

    # synthetic
    "save_synthetic_stream",

    # individual steps (optional)
    "scrape_data",
    "partition_articles",
]