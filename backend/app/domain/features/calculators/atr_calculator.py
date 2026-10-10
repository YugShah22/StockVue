from decimal import Decimal
from typing import Final

from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_calculator import FeatureCalculator
from app.domain.features.feature_definition import FeatureDefinition


class ATRCalculator(FeatureCalculator):
    """Calculates Average True Range (ATR) using Wilder's smoothing.

    True Range for bar t (t >= 1):
        TR_t = max(high_t - low_t, abs(high_t - close_(t-1)), abs(low_t - close_(t-1)))

    Initial ATR (window n):
        ATR_n = sum(first_n_true_ranges) / n

    Wilder's smoothing for subsequent bars:
        ATR_t = (ATR_(t-1) * (n - 1) + TR_t) / n

    Output is a non-negative Decimal in absolute price units.
    """

    FEATURE_NAME: Final[str] = "atr"
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

        # Requires at least n + 1 bars to produce n True Range values
        if len(context.market_bars) < window + 1:
            return None

        # Sort chronologically ascending without mutating context.market_bars
        sorted_bars = sorted(context.market_bars, key=lambda b: b.timestamp)

        tr_values: list[Decimal] = []
        for i in range(1, len(sorted_bars)):
            curr_bar = sorted_bars[i]
            prev_close = sorted_bars[i - 1].close

            tr_t = max(
                curr_bar.high - curr_bar.low,
                abs(curr_bar.high - prev_close),
                abs(curr_bar.low - prev_close),
            )
            tr_values.append(tr_t)

        window_dec = Decimal(window)

        # Initial ATR: arithmetic mean of the first n True Range values
        atr = sum(tr_values[:window], Decimal(0)) / window_dec

        # Wilder's smoothing for subsequent True Range values
        window_minus_one = Decimal(window - 1)
        for i in range(window, len(tr_values)):
            atr = (atr * window_minus_one + tr_values[i]) / window_dec

        return atr

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
