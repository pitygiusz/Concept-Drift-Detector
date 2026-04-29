from .basic_synthetic import main as basic_synthetic
from .advanced_synthetic import main as advanced_synthetic
from .baseline_balanced_real import main as baseline_balanced_real
from .baseline_extracted_real import main as baseline_extracted_real
from .transformer_extracted_real import main as transformer_extracted_real

__all__ = [
    "basic_synthetic",
    "advanced_synthetic",
    "baseline_balanced_real",
    "baseline_extracted_real",
    "transformer_extracted_real"
]

# Important note: when new experiments is created, it should be added here!