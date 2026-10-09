from decimal import Decimal
from typing import Final

from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_calculator import FeatureCalculator
from app.domain.features.feature_definition import FeatureDefinition


class ReturnCalculator(FeatureCalculator):
    """Calculates n-period price return.

    Formula:
        return = (latest_close / close_n_periods_ago) - 1

    Result is represented as a fractional Decimal (e.g. 0.05 for 5%).
    """

    FEATURE_NAME: Final[str] = "return"
    DEFAULT_WINDOW: Final[int] = 1

    @property
    def feature_name(self) -> str:
        return self.FEATURE_NAME

    def calculate(
        self,
        definition: FeatureDefinition,
        context: FeatureCalculationContext,
    ) -> Decimal | None:
        window = self._extract_window(definition)

        # Requires at least n + 1 bars (today + n periods ago)
        if len(context.market_bars) < window + 1:
            return None

        # Sort chronologically ascending without mutating context.market_bars
        sorted_bars = sorted(context.market_bars, key=lambda b: b.timestamp)

        latest_bar = sorted_bars[-1]
        comparison_bar = sorted_bars[-(window + 1)]

        comparison_close = comparison_bar.close
        if comparison_close <= Decimal(0):
            return None

        latest_close = latest_bar.close
        return (latest_close / comparison_close) - Decimal(1)

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
