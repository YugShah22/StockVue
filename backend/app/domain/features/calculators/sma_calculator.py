from decimal import Decimal
from typing import Final

from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_calculator import FeatureCalculator
from app.domain.features.feature_definition import FeatureDefinition


class SMACalculator(FeatureCalculator):
    """Calculates n-period Simple Moving Average (SMA).

    Formula:
        SMA = sum(last n closing prices) / n

    Result is in the same price units as the underlying close prices.
    """

    FEATURE_NAME: Final[str] = "sma"
    DEFAULT_WINDOW: Final[int] = 20

    @property
    def feature_name(self) -> str:
        return self.FEATURE_NAME

    def calculate(
        self,
        definition: FeatureDefinition,
        context: FeatureCalculationContext,
    ) -> Decimal | None:
        window = self._extract_window(definition)

        # Requires at least n bars
        if len(context.market_bars) < window:
            return None

        # Sort chronologically ascending without mutating context.market_bars
        sorted_bars = sorted(context.market_bars, key=lambda b: b.timestamp)

        window_bars = sorted_bars[-window:]
        total = sum((bar.close for bar in window_bars), Decimal(0))
        return total / Decimal(window)

    def _extract_window(self, definition: FeatureDefinition) -> int:
        if "window" not in definition.parameters:
            return self.DEFAULT_WINDOW

        window = definition.parameters["window"]
        if isinstance(window, bool) or not isinstance(window, int):
            raise TypeError(
                f"Parameter 'window' must be an integer, got {type(window).__name__}"
            )

        if window <= 0:
            raise ValueError(
                f"Parameter 'window' must be a positive integer, got {window}"
            )

        return window
