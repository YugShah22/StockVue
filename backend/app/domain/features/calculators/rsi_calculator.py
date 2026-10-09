from decimal import Decimal
from typing import Final

from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_calculator import FeatureCalculator
from app.domain.features.feature_definition import FeatureDefinition


class RSICalculator(FeatureCalculator):
    """Calculates Relative Strength Index (RSI) using Wilder's smoothing.

    Formulation:
        RSI = 100 * average_gain / (average_gain + average_loss)

    Edge cases:
        - average_gain == 0 and average_loss > 0: Decimal("0")
        - average_gain > 0 and average_loss == 0: Decimal("100")
        - average_gain == 0 and average_loss == 0: Decimal("50")

    Output is a Decimal strictly bounded in [0, 100].
    """

    FEATURE_NAME: Final[str] = "rsi"
    DEFAULT_WINDOW: Final[int] = 14

    @property
    def feature_name(self) -> str:
        return self.FEATURE_NAME

    def calculate(
        self,
        definition: FeatureDefinition,
        context: FeatureCalculationContext,
    ) -> Decimal | None:
        window = self._extract_window(definition)

        # Requires at least n + 1 bars to compute n initial price changes
        if len(context.market_bars) < window + 1:
            return None

        # Sort chronologically ascending without mutating context.market_bars
        sorted_bars = sorted(context.market_bars, key=lambda b: b.timestamp)
        closes = [bar.close for bar in sorted_bars]

        gains: list[Decimal] = []
        losses: list[Decimal] = []

        for i in range(1, len(closes)):
            delta = closes[i] - closes[i - 1]
            if delta > Decimal(0):
                gains.append(delta)
                losses.append(Decimal(0))
            elif delta < Decimal(0):
                gains.append(Decimal(0))
                losses.append(-delta)
            else:
                gains.append(Decimal(0))
                losses.append(Decimal(0))

        window_dec = Decimal(window)

        # Initial averages over first n price changes
        avg_gain = sum(gains[:window], Decimal(0)) / window_dec
        avg_loss = sum(losses[:window], Decimal(0)) / window_dec

        # Wilder's smoothing for subsequent price changes
        window_minus_one = Decimal(window - 1)
        for i in range(window, len(gains)):
            avg_gain = (avg_gain * window_minus_one + gains[i]) / window_dec
            avg_loss = (avg_loss * window_minus_one + losses[i]) / window_dec

        # Handle edge cases explicitly
        if avg_gain == Decimal(0) and avg_loss == Decimal(0):
            return Decimal("50")

        if avg_loss == Decimal(0):
            return Decimal("100")

        if avg_gain == Decimal(0):
            return Decimal("0")

        return (Decimal(100) * avg_gain) / (avg_gain + avg_loss)

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
