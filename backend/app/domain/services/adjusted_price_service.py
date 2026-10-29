"""Pure domain service for split-adjusted market-bar calculations."""

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from uuid import UUID

from app.domain.entities.corporate_action import CorporateAction
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.corporate_action_type import CorporateActionType


class AdjustedPriceService:
    """Pure domain service that computes split-adjusted market bars.

    Conventions and Semantics:
    --------------------------
    1. CorporateAction.execution_date:
       Interpreted as the effective market execution date (ex-date).
       - Bars on or after execution_date (``bar.timestamp.date() >= execution_date``)
         are traded on the post-split basis and do not have their prices or volume
         adjusted for this split.
       - Bars strictly prior to execution_date (``bar.timestamp.date() < execution_date``)
         are traded on the pre-split basis and are back-adjusted by the split factor.
    2. Split Ratio Direction (CorporateAction.value):
       Defined as ``new_shares / old_shares``:
       - A 2-for-1 forward split has ``value = Decimal("2")``.
         Historical OHLC prices are divided by 2; volume is multiplied by 2.
       - A 1-for-2 reverse split has ``value = Decimal("0.5")``.
         Historical OHLC prices are divided by 0.5 (doubled); volume is multiplied by 0.5 (halved).
       - The identical factor is applied to Open, High, Low, and Close.
    3. Multiple Splits:
       When multiple split corporate actions apply to a historical bar, their
       factors combine multiplicatively:
       ``cumulative_factor = product(split.value for split in matching_splits)``.
    4. Point-in-Time Isolation (as_of_date):
       If ``as_of_date`` is provided, only corporate actions with
       ``execution_date <= as_of_date`` are considered effective. Any action with
       ``execution_date > as_of_date`` is ignored to prevent future look-ahead leakage.
    5. Raw-Bar Immutability:
       Input ``MarketBar`` instances are never mutated. New ``MarketBar`` instances
       are instantiated and returned.
    6. Non-Split Actions:
       Corporate actions of types other than SPLIT (e.g. DIVIDEND) are ignored
       during OHLCV price/volume adjustment.
    7. Deterministic and Chronological Output:
       Returned bars are sorted chronologically by (instrument_id, timestamp).
    """

    @staticmethod
    def adjust(
        bars: Sequence[MarketBar],
        corporate_actions: Sequence[CorporateAction],
        as_of_date: date | None = None,
    ) -> list[MarketBar]:
        """Return split-adjusted market bars for the given sequence of bars.

        Parameters
        ----------
        bars:
            Sequence of raw MarketBar instances to adjust.
        corporate_actions:
            Sequence of CorporateAction entities (splits, dividends, etc.).
        as_of_date:
            Optional point-in-time cutoff date. If supplied, any corporate action
            with execution_date > as_of_date is excluded.

        Returns
        -------
        list[MarketBar]
            New MarketBar instances with split-adjusted OHLC and volume,
            sorted chronologically by (instrument_id, timestamp).
        """
        if not bars:
            return []

        # Filter to SPLIT corporate actions effective as of as_of_date
        effective_splits: list[CorporateAction] = []
        for action in corporate_actions:
            if action.action_type != CorporateActionType.SPLIT:
                continue
            if as_of_date is not None and action.execution_date > as_of_date:
                continue
            if (
                not isinstance(action.value, Decimal)
                or not action.value.is_finite()
                or action.value <= Decimal("0")
            ):
                raise ValueError(
                    f"Corporate action {action.corporate_action_id} has "
                    f"invalid split ratio {action.value}"
                )
            effective_splits.append(action)

        # Index splits by instrument_id for efficient lookup
        splits_by_instrument: dict[UUID, list[CorporateAction]] = {}
        for split in effective_splits:
            splits_by_instrument.setdefault(split.instrument_id, []).append(split)

        # Sort input bars chronologically by (instrument_id, timestamp)
        sorted_bars = sorted(bars, key=lambda b: (b.instrument_id, b.timestamp))

        adjusted_bars: list[MarketBar] = []
        for bar in sorted_bars:
            bar_splits = splits_by_instrument.get(bar.instrument_id, [])
            bar_date = bar.timestamp.date()

            # A split back-adjusts bars strictly prior to its execution date (ex-date)
            applicable_splits = [s for s in bar_splits if bar_date < s.execution_date]

            if not applicable_splits:
                # No splits apply to this bar; return a new instance with identical values
                adjusted_bars.append(
                    MarketBar(
                        instrument_id=bar.instrument_id,
                        timestamp=bar.timestamp,
                        open=bar.open,
                        high=bar.high,
                        low=bar.low,
                        close=bar.close,
                        volume=bar.volume,
                        market_bar_id=bar.market_bar_id,
                    )
                )
                continue

            # Calculate cumulative split factor
            cumulative_factor = Decimal("1")
            for split in applicable_splits:
                cumulative_factor *= split.value

            adjusted_bars.append(
                MarketBar(
                    instrument_id=bar.instrument_id,
                    timestamp=bar.timestamp,
                    open=bar.open / cumulative_factor,
                    high=bar.high / cumulative_factor,
                    low=bar.low / cumulative_factor,
                    close=bar.close / cumulative_factor,
                    volume=bar.volume * cumulative_factor,
                    market_bar_id=bar.market_bar_id,
                )
            )

        return adjusted_bars
