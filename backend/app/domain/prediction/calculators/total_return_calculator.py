"""Calculates split-aware, dividend-aware forward total return for machine learning targets."""

from collections.abc import Sequence
from datetime import date
from decimal import Decimal
from uuid import UUID

from app.domain.entities.corporate_action import CorporateAction
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.corporate_action_type import CorporateActionType
from app.domain.prediction.calculators.forward_return_calculator import (
    ForwardReturnCalculator,
    _bar_date,
)
from app.domain.prediction.target_definition import TargetDefinition
from app.domain.prediction.target_value import TargetValue
from app.domain.services.adjusted_price_service import AdjustedPriceService


class TotalReturnCalculator(ForwardReturnCalculator):
    """Calculates forward cumulative total return, accounting for splits and dividends.

    Total return accounts for:
    1. Stock splits:
       - Endpoint prices P(t) and P(t+h) are converted to a consistent share basis
         as of realization_date (t+h) via ``AdjustedPriceService``.
       - Any stock split effective after realization_date is excluded (point-in-time).
    2. Cash dividends:
       - Dividends with execution_date in (observation_date, realization_date] are
         reinvested at the raw closing price of the ex-dividend date.
       - Reinvestment factor for each ex-date: F_d = 1 + (D_raw / P_exdiv_raw).
       - Consistent share basis: evaluating D_raw and P_exdiv_raw on the raw ex-date
         basis is scale-invariant to subsequent splits and avoids distorted dividend
         yields.
    3. Multi-period compounding:
       - (1 + R_total) = (P_adj(t+h) / P_adj(t)) * product(1 + D_k / P_k_close)
       - R_total = (P_adj(t+h) / P_adj(t) * product(F_d)) - 1.

    Rules and contract:
    - Strict half-open interval: actions on observation_date are excluded; actions
      on realization_date are included.
    - If a dividend ex-date has no matching bar for the instrument, returns None
      (unrealizable target without fabricating prices).
    - If any relevant close price is zero or negative, returns None.
    - Zero mutation of caller's input bars.
    - Pure Decimal financial math throughout.
    """

    def __init__(self, corporate_actions: Sequence[CorporateAction] = ()) -> None:
        self._corporate_actions = tuple(corporate_actions)

    @property
    def corporate_actions(self) -> tuple[CorporateAction, ...]:
        return self._corporate_actions

    def calculate(
        self,
        definition: TargetDefinition,
        instrument_id: UUID,
        bars: Sequence[MarketBar],
        observation_date: date,
        corporate_actions: Sequence[CorporateAction] | None = None,
    ) -> TargetValue | None:
        """Calculate forward total return.

        Parameters
        ----------
        definition:
            The target definition (typically named 'total_return' or 'forward_return').
        instrument_id:
            UUID of the instrument being evaluated.
        bars:
            Available market bars.
        observation_date:
            The calendar date of the observation bar (P_t).
        corporate_actions:
            Optional sequence of corporate actions overriding instance-level actions.

        Returns
        -------
        TargetValue | None
            The computed total return, or None if unrealizable.
        """
        # 1. Restrict calculation strictly to the requested instrument
        instrument_bars = [b for b in bars if b.instrument_id == instrument_id]

        # 2. Sort chronologically ascending
        sorted_bars = sorted(instrument_bars, key=lambda b: b.timestamp)

        # 3. Locate observation bar
        obs_bars = [
            (idx, bar) for idx, bar in enumerate(sorted_bars) if _bar_date(bar) == observation_date
        ]

        if len(obs_bars) > 1:
            raise ValueError(
                f"TotalReturnCalculator: {len(obs_bars)} bars found for "
                f"observation_date {observation_date} (instrument {instrument_id}); "
                f"expected exactly 1. Deduplicate the input before calling calculate()."
            )

        if not obs_bars:
            return None

        obs_idx, obs_bar = obs_bars[0]
        if obs_bar.close <= Decimal(0):
            return None

        # 4. Check sufficient future bars
        h = definition.horizon_bars
        future_idx = obs_idx + h
        if future_idx >= len(sorted_bars):
            return None

        future_bar = sorted_bars[future_idx]
        realization_date = _bar_date(future_bar)

        if realization_date <= observation_date:
            return None

        # 5. Resolve corporate actions for this instrument
        actions_to_use = (
            corporate_actions if corporate_actions is not None else self._corporate_actions
        )
        inst_actions = [a for a in actions_to_use if a.instrument_id == instrument_id]

        # 6. Split adjustment: adjust endpoint prices to realization_date basis
        split_actions = [
            a
            for a in inst_actions
            if a.action_type == CorporateActionType.SPLIT and a.execution_date <= realization_date
        ]

        adj_endpoints = AdjustedPriceService.adjust(
            [obs_bar, future_bar],
            split_actions,
            as_of_date=realization_date,
        )

        adj_obs = next((b for b in adj_endpoints if _bar_date(b) == observation_date), None)
        adj_future = next((b for b in adj_endpoints if _bar_date(b) == realization_date), None)

        if adj_obs is None or adj_future is None:
            return None

        if adj_obs.close <= Decimal(0):
            return None

        price_ratio = adj_future.close / adj_obs.close

        # 7. Dividend reinvestment: dividends in (observation_date, realization_date]
        div_actions = [
            a
            for a in inst_actions
            if a.action_type == CorporateActionType.DIVIDEND
            and observation_date < a.execution_date <= realization_date
        ]

        # Group dividends by execution date in case of multiple distributions on same day
        divs_by_date: dict[date, Decimal] = {}
        for div in div_actions:
            divs_by_date[div.execution_date] = (
                divs_by_date.get(div.execution_date, Decimal(0)) + div.value
            )

        # Date-to-bar lookup for raw ex-dividend closing prices
        date_to_bar: dict[date, MarketBar] = {_bar_date(b): b for b in sorted_bars}

        cumulative_div_factor = Decimal(1)
        for ex_date in sorted(divs_by_date.keys()):
            ex_bar = date_to_bar.get(ex_date)
            if ex_bar is None or ex_bar.close <= Decimal(0):
                # Ex-date bar missing or invalid close price -> target cannot be realized
                return None

            raw_div_amount = divs_by_date[ex_date]
            reinvestment_factor = Decimal(1) + (raw_div_amount / ex_bar.close)
            cumulative_div_factor *= reinvestment_factor

        total_return = (price_ratio * cumulative_div_factor) - Decimal(1)

        return TargetValue(
            target=definition,
            instrument_id=instrument_id,
            observation_date=observation_date,
            value=total_return,
            realization_date=realization_date,
        )
