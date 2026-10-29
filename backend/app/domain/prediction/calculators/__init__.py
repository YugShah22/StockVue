"""Prediction target calculators package."""

from app.domain.prediction.calculators.forward_return_calculator import (
    ForwardReturnCalculator,
)
from app.domain.prediction.calculators.total_return_calculator import (
    TotalReturnCalculator,
)

__all__ = [
    "ForwardReturnCalculator",
    "TotalReturnCalculator",
]
