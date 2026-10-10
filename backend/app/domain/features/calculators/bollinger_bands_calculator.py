from decimal import Decimal
from typing import Final

from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_calculator import FeatureCalculator
from app.domain.features.feature_definition import FeatureDefinition


class BollingerBandsCalculator(FeatureCalculator):
    """Calculates Bollinger Bands for a closing price series.

    Formulas:
        mean = sum(closes) / window
        variance = sum((price - mean) ** 2 for price in closes) / window  (ddof=0)
        std_dev = variance.sqrt()

        "middle": mean
        "upper":  mean + (k * std_dev)
        "lower":  mean - (k * std_dev)

    Supported parameters:
        - window: positive integer, default 20.
        - k: positive integer or Decimal, default Decimal("2").
        - band: one of "upper", "middle", "lower", default "middle".
    """

    FEATURE_NAME: Final[str] = "bollinger_bands"
    DEFAULT_WINDOW: Final[int] = 20
    DEFAULT_K: Final[Decimal] = Decimal("2")
    DEFAULT_BAND: Final[str] = "middle"
    VALID_BANDS: Final[frozenset[str]] = frozenset({"upper", "middle", "lower"})

    @property
    def feature_name(self) -> str:
        return self.FEATURE_NAME

    def calculate(
        self,
        definition: FeatureDefinition,
        context: FeatureCalculationContext,
    ) -> Decimal | None:
        window = self._extract_window(definition)
        k = self._extract_k(definition)
        band = self._extract_band(definition)

        # Requires at least window bars
        if len(context.market_bars) < window:
            return None

        # Sort chronologically ascending without mutating context.market_bars
        sorted_bars = sorted(context.market_bars, key=lambda b: b.timestamp)
        window_bars = sorted_bars[-window:]
        closes = [bar.close for bar in window_bars]

        window_dec = Decimal(window)
        mean = sum(closes, Decimal(0)) / window_dec

        variance = sum(((price - mean) ** 2 for price in closes), Decimal(0)) / window_dec
        std_dev = variance.sqrt()

        if band == "middle":
            return mean
        if band == "upper":
            return mean + (k * std_dev)
        if band == "lower":
            return mean - (k * std_dev)

        raise ValueError(f"Unsupported band: '{band}'")

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

    def _extract_k(self, definition: FeatureDefinition) -> Decimal:
        if "k" not in definition.parameters:
            return self.DEFAULT_K

        k = definition.parameters["k"]
        if isinstance(k, bool):
            raise TypeError(
                f"Parameter 'k' must be an integer or Decimal, got {type(k).__name__}"
            )

        if isinstance(k, int):
            if k <= 0:
                raise ValueError(
                    f"Parameter 'k' must be positive, got {k}"
                )
            return Decimal(k)

        if isinstance(k, Decimal):
            if not k.is_finite():
                raise ValueError(
                    f"Parameter 'k' must be a finite Decimal, got {k}"
                )
            if k <= Decimal(0):
                raise ValueError(
                    f"Parameter 'k' must be positive, got {k}"
                )
            return k

        raise TypeError(
            f"Parameter 'k' must be an integer or Decimal, got {type(k).__name__}"
        )

    def _extract_band(self, definition: FeatureDefinition) -> str:
        if "band" not in definition.parameters:
            return self.DEFAULT_BAND

        band = definition.parameters["band"]
        if not isinstance(band, str):
            raise TypeError(
                f"Parameter 'band' must be a string, got {type(band).__name__}"
            )

        if band not in self.VALID_BANDS:
            raise ValueError(
                f"Parameter 'band' must be one of {sorted(self.VALID_BANDS)}, got '{band}'"
            )

        return band
