from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from uuid import UUID

from app.domain.entities.market_bar import MarketBar
from app.domain.prediction.target_definition import TargetDefinition
from app.domain.prediction.target_value import TargetValue


class ForwardReturnCalculator:
    """Calculates the h-bar forward cumulative return for a given observation date.

    Formula::

        R(t, t+h) = P(t+h) / P(t) - 1

    where:
    - ``P(t)``   is the close of the observation bar (the bar whose date matches
                 ``observation_date``).
    - ``P(t+h)`` is the close of the bar exactly ``h`` trading bars *after* the
                 observation bar in the chronologically sorted sequence of the
                 requested instrument.
    - ``h``      is ``definition.horizon_bars``.

    Rules / contract
    ----------------
    1. Input bars are filtered by ``instrument_id`` *before* any sorting,
       observation-date matching, duplicate checking, or horizon counting. Bars
       belonging to other instruments are completely ignored and never count
       towards the forward horizon. The caller's input sequence is never mutated.
    2. Filtered bars are sorted chronologically (ascending timestamp).
    3. Observation-bar resolution uses the calendar date of the bar's timestamp
       (``bar.timestamp.date()``), preserving the exchange-local or attached
       market date without artificial conversion to UTC.
    4. If more than one bar for the requested instrument maps to ``observation_date``
       (duplicate records), a ``ValueError`` is raised immediately. Bars for
       other instruments on the observation date do not trigger this error.
    5. Returns ``None`` when:
       - No bar exists for ``observation_date`` for the requested instrument.
       - Fewer than ``h`` bars of the requested instrument follow the observation
         bar (insufficient future history).
       - The observation close is zero or negative (matching ``ReturnCalculator``).
    6. Never accesses repositories, the Feature Engine, databases, APIs or any
       external service.
    7. Pure ``Decimal`` arithmetic throughout; no float conversion.
    """

    def calculate(
        self,
        definition: TargetDefinition,
        instrument_id: UUID,
        bars: Sequence[MarketBar],
        observation_date: date,
    ) -> TargetValue | None:
        """Calculate the forward return target value.

        Parameters
        ----------
        definition:
            The target definition carrying ``horizon_bars`` and metadata.
        instrument_id:
            UUID of the instrument being evaluated. The calculation is strictly
            restricted to bars matching this instrument ID.
        bars:
            Available market bars. May contain bars from multiple instruments,
            may be unsorted, and may include bars both before and after
            ``observation_date``.
        observation_date:
            The calendar date of the observation bar (``P_t``).

        Returns
        -------
        TargetValue | None
            The computed fractional return, or ``None`` if the target cannot be
            realised (see class docstring for all conditions).

        Raises
        ------
        ValueError
            If more than one bar for the requested instrument maps to
            ``observation_date`` (ambiguous input).
        """
        # 1. Restrict calculation strictly to the requested instrument FIRST
        instrument_bars = [b for b in bars if b.instrument_id == instrument_id]

        # 2. Sort chronologically ascending without mutating caller's sequence
        sorted_bars = sorted(instrument_bars, key=lambda b: b.timestamp)

        # 3. Locate bars matching the observation date
        obs_bars: list[tuple[int, MarketBar]] = [
            (idx, bar) for idx, bar in enumerate(sorted_bars) if _bar_date(bar) == observation_date
        ]

        # 4. Reject ambiguous duplicates for the requested instrument
        if len(obs_bars) > 1:
            raise ValueError(
                f"ForwardReturnCalculator: {len(obs_bars)} bars found for "
                f"observation_date {observation_date} (instrument {instrument_id}); "
                f"expected exactly 1. Deduplicate the input before calling calculate()."
            )

        # 5. Missing observation bar for this instrument → None
        if not obs_bars:
            return None

        obs_idx, obs_bar = obs_bars[0]
        observation_close = obs_bar.close

        # 6. Non-positive observation close → None (mirrors ReturnCalculator)
        if observation_close <= Decimal(0):
            return None

        # 7. Check sufficient future bars (counted strictly within this instrument)
        h = definition.horizon_bars
        future_idx = obs_idx + h
        if future_idx >= len(sorted_bars):
            return None

        future_bar = sorted_bars[future_idx]

        # 8. Sanity guard: future bar must be strictly after the observation bar
        if _bar_date(future_bar) <= observation_date:
            return None

        future_close = future_bar.close
        forward_return = (future_close / observation_close) - Decimal(1)

        return TargetValue(
            target=definition,
            instrument_id=instrument_id,
            observation_date=observation_date,
            value=forward_return,
            realization_date=_bar_date(future_bar),
        )


def _bar_date(bar: MarketBar) -> date:
    """Return the calendar market date for a bar's timestamp.

    Uses ``bar.timestamp.date()`` directly, preserving the bar's native
    market date without artificial conversion to UTC.
    """
    return bar.timestamp.date()
