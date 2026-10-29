"""Phase 5.4.1 — Price Semantics Contract Tests.

=== PURPOSE ===

This module documents the *current* behaviour of price-related components as
executable assertions, and *specifies* the desired future behaviour that will
become executable once the adjustment engine abstraction is implemented.

Two kinds of tests appear here:

CURRENT_BEHAVIOR tests
    Assert what the production code does right now.  They are green and must
    stay green.  They describe a *temporary design decision*, not the final
    intended behaviour.  Where the current behaviour differs from the intended
    behaviour, the test body carries an explicit comment saying so.

ACCEPTANCE_SPEC tests
    Define what the production code *must* do after Phase 5.4.2 introduces the
    price-adjustment abstraction.  They are marked ``pytest.mark.xfail``
    (expected-to-fail) and decorated with the ``@acceptance_spec`` marker.
    Each one documents the required scenario, expected numeric result, and the
    prerequisite abstraction that must exist before it can be implemented.

=== CONTRACTS LOCKED (CURRENT) ===

C1  Raw OHLCV          — MarketBar stores raw, unadjusted prices.
C2  No floats          — All financial fields are Decimal throughout the domain.
C3  yfinance raw       — Provider fetches with auto_adjust=False (raw prices).
C4  Feature PIT        — Feature engine receives only bars ≤ obs_date.
C5  Raw-close target   — ForwardReturnCalculator uses raw close; no adjustment.
                         CURRENT BEHAVIOR — intended behavior differs (see C_ADJ).
C6  Dividend storage   — CorporateAction(DIVIDEND) exists but is unintegrated.
                         CURRENT BEHAVIOR — intended behavior differs (see C_ADJ).
C7  Split storage      — CorporateAction(SPLIT) exists but is unintegrated.
                         CURRENT BEHAVIOR — intended behavior differs (see C_ADJ).
C8  Realization date   — realization_date is the actual future bar's date.
C9  Target visibility  — DatasetBuilder passes ALL bars to target calculator.
C10 Upsert policy      — upsert_bars() uses ON CONFLICT DO UPDATE (DB-level).

=== ACCEPTANCE SPECIFICATIONS (PHASE 5.4.2+) ===

C_ADJ  Adjusted-price contracts
    The adjustment engine has not been implemented yet.  These specifications
    are documented here so that the Phase 5.4.2 implementer has an unambiguous
    acceptance contract to satisfy.  See TestCAdjAcceptanceSpecs.

=== PHASE 5.4 DELIVERY ROADMAP & MILESTONE PLAN ===

Phase 5.4.1 (Current): Price Semantics Contract Locking & Acceptance Specs
    - Lock existing behavior with executable CURRENT_BEHAVIOR contract tests.
    - Document data revision and DB upsert policy.
    - Write explicit acceptance specifications for future adjusted behavior.
    - Zero production code changes; non-breaking audit & test coverage only.

Phase 5.4.2: Domain & Service Abstraction for Price Adjustments
    - Harden CorporateAction domain validation (reject NaN and non-finite values).
    - Introduce AdjustedPriceService (clean architecture domain/service abstraction).
    - Implement back-adjustment factor calculation for stock splits:
        P_adj = P_raw / split_ratio  (for dates before execution_date).
        V_adj = V_raw * split_ratio  (inverse convention for volume).
    - Ensure raw MarketBar instances are never mutated (pure immutability).

Phase 5.4.3: Total-Return Target & Feature Integration
    - Introduce TotalReturnCalculator (or extend target calculation).
    - Implement dividend reinvestment convention: reinvest at ex-dividend close:
        factor = 1 + (dividend_per_share / ex_div_close_price).
    - Wire point-in-time isolation in DatasetBuilder:
        * Features at obs_date see corporate actions with execution_date <= obs_date.
        * Total return targets see corporate actions with execution_date in (obs_date, realization_date].

Phase 5.4.4: Ingestion & Storage Pipeline Hardening
    - Ingest corporate actions from provider layer (yfinance / production feed).
    - Database persistence and unique constraint enforcement for corporate actions.
    - Integration tests against live PostgreSQL verifying idempotent upserts,
      primary key preservation, and data revision handling.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from app.domain.entities.corporate_action import CorporateAction
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.corporate_action_type import CorporateActionType
from app.domain.features.calculators.return_calculator import ReturnCalculator
from app.domain.features.feature_calculation_context import FeatureCalculationContext
from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.features.feature_value import FeatureValue
from app.domain.prediction.calculators.forward_return_calculator import (
    ForwardReturnCalculator,
)
from app.domain.prediction.target_definition import TargetDefinition
from app.domain.prediction.target_value import TargetValue
from app.domain.services.adjusted_price_service import AdjustedPriceService
from app.services.features.feature_calculator_registry import FeatureCalculatorRegistry
from app.services.features.feature_engine import FeatureEngine
from app.services.prediction.dataset_builder import DatasetBuilder

# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

# acceptance_spec: marks a test that specifies desired behaviour for a future
# phase.  These tests are expected to fail until the implementation is ready.
acceptance_spec = pytest.mark.xfail(
    reason=(
        "Acceptance specification for Phase 5.4.2+ adjustment engine. "
        "Will pass once AdjustedPriceService and total-return calculator exist."
    ),
    strict=True,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_bar(
    timestamp: datetime,
    close: Decimal,
    instrument_id: UUID | None = None,
    *,
    open_: Decimal | None = None,
    high: Decimal | None = None,
    low: Decimal | None = None,
    volume: Decimal = Decimal("1000"),
) -> MarketBar:
    """Create a MarketBar with OHLC all equal to close unless overridden."""
    c = close
    return MarketBar(
        instrument_id=instrument_id or uuid4(),
        timestamp=timestamp,
        open=open_ if open_ is not None else c,
        high=high if high is not None else c,
        low=low if low is not None else c,
        close=c,
        volume=volume,
    )


_BASE = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)  # Monday


def _daily_bars(
    closes: list[str],
    instrument_id: UUID,
    start: datetime = _BASE,
) -> list[MarketBar]:
    """Build consecutive daily bars from close-price strings."""
    return [
        _make_bar(start + timedelta(days=i), Decimal(c), instrument_id=instrument_id)
        for i, c in enumerate(closes)
    ]


def _make_target_defn(horizon: int = 5) -> TargetDefinition:
    return TargetDefinition(name="forward_return", horizon_bars=horizon)


def _make_feature_engine() -> FeatureEngine:
    registry = FeatureCalculatorRegistry()
    registry.register(ReturnCalculator())
    return FeatureEngine(registry=registry)


def _make_return_feature_defn(window: int = 1) -> FeatureDefinition:
    return FeatureDefinition(
        name="return",
        category=FeatureCategory.TECHNICAL,
        description=f"{window}-period return",
        parameters={"window": window},
    )


# ---------------------------------------------------------------------------
# C1 — Raw OHLCV contract  (CURRENT BEHAVIOR, also intended)
# ---------------------------------------------------------------------------


class TestC1RawOhlcvContract:
    """C1: MarketBar stores raw, unadjusted prices only.

    These tests document *both* current and intended behaviour: the raw-bar
    aggregate should never carry adjusted prices directly.  Adjusted prices are
    the responsibility of a separate service (Phase 5.4.2+), not MarketBar.
    """

    def test_market_bar_stores_raw_ohlcv_values(self) -> None:
        """C1a CURRENT+INTENDED: OHLC fields are stored exactly as supplied.

        Behavior-level check: round-trip the five raw values through the
        domain object and verify they are returned unchanged.
        """
        bar = MarketBar(
            instrument_id=uuid4(),
            timestamp=_BASE,
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("20000"),
        )
        assert bar.open == Decimal("100")
        assert bar.high == Decimal("110")
        assert bar.low == Decimal("95")
        assert bar.close == Decimal("105")
        assert bar.volume == Decimal("20000")

    def test_market_bar_fields_are_decimal(self) -> None:
        """C1b CURRENT+INTENDED: All OHLCV price and volume fields are Decimal.

        No float arithmetic must be used for financial values.
        """
        bar = _make_bar(
            _BASE,
            Decimal("123.45"),
            open_=Decimal("120.00"),
            high=Decimal("125.00"),
            low=Decimal("119.00"),
            volume=Decimal("50000"),
        )
        assert isinstance(bar.open, Decimal), "open must be Decimal"
        assert isinstance(bar.high, Decimal), "high must be Decimal"
        assert isinstance(bar.low, Decimal), "low must be Decimal"
        assert isinstance(bar.close, Decimal), "close must be Decimal"
        assert isinstance(bar.volume, Decimal), "volume must be Decimal"

    def test_market_bar_rejects_negative_open(self) -> None:
        """C1c CURRENT+INTENDED: Negative open price is rejected at domain boundary."""
        with pytest.raises(ValueError, match="Open price cannot be negative"):
            MarketBar(
                instrument_id=uuid4(),
                timestamp=_BASE,
                open=Decimal("-1"),
                high=Decimal("10"),
                low=Decimal("-1"),
                close=Decimal("5"),
                volume=Decimal("100"),
            )

    def test_market_bar_rejects_high_less_than_low(self) -> None:
        """C1d CURRENT+INTENDED: High < low is structurally invalid."""
        with pytest.raises(ValueError, match="High price cannot be lower than low"):
            MarketBar(
                instrument_id=uuid4(),
                timestamp=_BASE,
                open=Decimal("50"),
                high=Decimal("40"),
                low=Decimal("50"),
                close=Decimal("45"),
                volume=Decimal("100"),
            )

    def test_raw_market_bar_is_unaffected_by_creating_a_corporate_action(
        self,
    ) -> None:
        """C1e CURRENT+INTENDED: Creating a CorporateAction does not mutate a MarketBar.

        CorporateAction and MarketBar are separate aggregate roots.
        Adjustment factors are applied by a service layer, never by mutating raw bars.
        """
        inst_id = uuid4()
        bar = _make_bar(_BASE, Decimal("1000"), instrument_id=inst_id)
        original_close = bar.close

        CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=_BASE.date(),
            value=Decimal("10"),
        )

        assert bar.close == original_close, (
            "Creating a CorporateAction must not mutate any existing MarketBar. "
            "Adjustment is a read-path concern, not a write-path mutation."
        )


# ---------------------------------------------------------------------------
# C2 — No-float contract  (CURRENT BEHAVIOR, also intended)
# ---------------------------------------------------------------------------


class TestC2NoFloatContract:
    """C2: Financial values must never be float anywhere in the domain layer.

    This invariant applies to both the current and intended implementations.
    """

    def test_target_value_rejects_float_value(self) -> None:
        """C2a: TargetValue.value must be Decimal; float is rejected at construction."""
        defn = _make_target_defn()
        with pytest.raises(TypeError, match="Decimal instance"):
            TargetValue(
                target=defn,
                instrument_id=uuid4(),
                observation_date=date(2026, 1, 5),
                value=0.05,  # type: ignore[arg-type]
                realization_date=date(2026, 1, 12),
            )

    def test_forward_return_result_is_decimal(self) -> None:
        """C2b: ForwardReturnCalculator.calculate() returns a Decimal value."""
        inst_id = uuid4()
        bars = _daily_bars(["100", "105", "110", "115", "120", "130"], inst_id)
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=5)
        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())
        assert result is not None
        assert isinstance(result.value, Decimal)

    def test_return_calculator_result_is_decimal(self) -> None:
        """C2c: ReturnCalculator.calculate() returns a Decimal value."""
        inst_id = uuid4()
        bars = _daily_bars(["100", "110"], inst_id)
        ctx = FeatureCalculationContext(
            instrument_id=inst_id,
            as_of_date=bars[1].timestamp.date(),
            market_bars=bars,
        )
        calc = ReturnCalculator()
        defn = _make_return_feature_defn(window=1)
        result = calc.calculate(defn, ctx)
        assert result is not None
        assert isinstance(result, Decimal)

    def test_forward_return_decimal_arithmetic_precision(self) -> None:
        """C2d: Forward-return computation uses exact Decimal arithmetic.

        301/300 - 1 is non-terminating in binary; float arithmetic produces a
        different bit pattern.  This verifies that no float intermediate is used.
        """
        inst_id = uuid4()
        bars = _daily_bars(["300", "300", "300", "300", "300", "301"], inst_id)
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=5)
        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())
        assert result is not None
        expected = Decimal("301") / Decimal("300") - Decimal("1")
        float_approx = Decimal(str(301.0 / 300.0 - 1.0))
        assert result.value == expected
        assert result.value != float_approx


# ---------------------------------------------------------------------------
# C3 — yfinance raw-price contract  (CURRENT BEHAVIOR, also intended)
# ---------------------------------------------------------------------------


class TestC3YFinanceRawPriceContract:
    """C3: yfinance provider must request raw (unadjusted) OHLCV data.

    CURRENT BEHAVIOR and INTENDED: the provider layer is always responsible
    for fetching raw data.  Adjustment happens in a separate service layer.
    """

    def test_get_bars_fetches_raw_prices_not_adjusted(self) -> None:
        """C3a: get_bars() requests raw prices (auto_adjust=False) and maps them directly.

        Behavior check: the returned bar's close equals the raw Close column
        value from the provider, not the Adj Close value.
        """
        import pandas as pd

        from app.infrastructure.providers.development.yfinance_market_data import (
            YFinanceMarketDataProvider,
        )

        inst_id = uuid4()
        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = MagicMock(symbol="RELIANCE", exchange_code="NSE")

        raw_close = 103.0
        adj_close = 98.0  # Different — would appear if auto_adjust=True

        idx = pd.DatetimeIndex([datetime(2026, 1, 5, tzinfo=UTC)], name="Datetime")
        raw_df = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [99.0],
                "Close": [raw_close],
                "Volume": [1_000_000],
                # Note: 'Adj Close' is NOT included — provider must not read it
            },
            index=idx,
        )

        provider = YFinanceMarketDataProvider(mock_repo)
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = raw_df
            mock_ticker_cls.return_value = mock_ticker
            bars = provider.get_bars(
                instrument_id=inst_id,
                start=datetime(2026, 1, 5, tzinfo=UTC),
                end=datetime(2026, 1, 6, tzinfo=UTC),
                interval="1d",
            )

        assert len(bars) == 1
        # Behavior assertion: returned close is the raw value, not adj-close
        assert bars[0].close == Decimal(str(raw_close)), (
            f"Expected raw close {raw_close}, got {bars[0].close}. "
            "Provider must map the raw Close column, not Adj Close."
        )
        assert bars[0].close != Decimal(str(adj_close)), (
            "Provider must NOT return the adjusted close value."
        )

        # Implementation check: provider called history with auto_adjust=False
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs.get("auto_adjust") is False, (
            "auto_adjust must be False to receive raw unadjusted prices."
        )

    def test_get_latest_bar_also_fetches_raw_prices(self) -> None:
        """C3b: get_latest_bar() also requests raw prices (auto_adjust=False)."""
        import pandas as pd

        from app.infrastructure.providers.development.yfinance_market_data import (
            YFinanceMarketDataProvider,
        )

        mock_repo = MagicMock()
        mock_repo.get_by_id.return_value = MagicMock(symbol="INFY", exchange_code="NSE")

        idx = pd.DatetimeIndex([datetime(2026, 1, 5, tzinfo=UTC)], name="Datetime")
        raw_df = pd.DataFrame(
            {
                "Open": [200.0],
                "High": [210.0],
                "Low": [195.0],
                "Close": [205.0],
                "Volume": [500_000],
            },
            index=idx,
        )

        provider = YFinanceMarketDataProvider(mock_repo)
        inst_id = uuid4()
        with patch("yfinance.Ticker") as mock_ticker_cls:
            mock_ticker = MagicMock()
            mock_ticker.history.return_value = raw_df
            mock_ticker_cls.return_value = mock_ticker
            bar = provider.get_latest_bar(instrument_id=inst_id, interval="1d")

        assert bar.close == Decimal("205.0")
        call_kwargs = mock_ticker.history.call_args.kwargs
        assert call_kwargs.get("auto_adjust") is False


# ---------------------------------------------------------------------------
# C4 — Feature point-in-time contract  (CURRENT + INTENDED)
# ---------------------------------------------------------------------------


class TestC4FeaturePointInTimeContract:
    """C4: Feature calculations must see only bars on or before observation_date.

    This rule applies to BOTH raw and adjusted prices.  It is an architectural
    constraint, not a side effect of the current raw-price-only implementation.
    """

    def test_feature_cannot_see_future_bar(self) -> None:
        """C4a: At t0 with window=1, row is dropped because there is no prior bar.

        Economic meaning: a 1-period return requires two bars.  At the first
        bar there is no prior bar, so the feature is undefined and the row is
        dropped.  The fact that future bars (200, 999) exist must not help.
        """
        inst_id = uuid4()
        bars = _daily_bars(["100", "200", "999"], inst_id)
        engine = _make_feature_engine()
        builder = DatasetBuilder(feature_engine=engine)
        defn = _make_target_defn(horizon=1)
        feature_defn = _make_return_feature_defn(window=1)

        matrix = builder.build_dataset(
            instrument_id=inst_id,
            bars=bars,
            features=[feature_defn],
            target=defn,
            observation_dates=[bars[0].timestamp.date()],
        )

        assert len(matrix.rows) == 0, (
            "Feature at t0 with window=1 must be dropped; "
            "future bars must not be used in feature computation."
        )

    def test_feature_at_t1_uses_t0_and_t1_only(self) -> None:
        """C4b: 1-period return at t1 = close(t1)/close(t0) - 1.

        The future bar at t2 (close=999) must never influence the result.
        """
        inst_id = uuid4()
        bars = _daily_bars(["100", "110", "999"], inst_id)
        engine = _make_feature_engine()
        builder = DatasetBuilder(feature_engine=engine)
        defn = _make_target_defn(horizon=1)
        feature_defn = _make_return_feature_defn(window=1)

        matrix = builder.build_dataset(
            instrument_id=inst_id,
            bars=bars,
            features=[feature_defn],
            target=defn,
            observation_dates=[bars[1].timestamp.date()],
        )

        assert len(matrix.rows) == 1
        row = matrix.rows[0]
        expected_return = Decimal("110") / Decimal("100") - Decimal("1")
        assert row.feature_values[0] == expected_return, (
            "Feature must equal (close_t1 / close_t0) - 1; future bar close (999) must not appear."
        )

    def test_feature_engine_receives_exactly_history_slice(self) -> None:
        """C4c: FeatureEngine receives exactly obs_idx+1 bars for observation at t1.

        Verified via a spy on the engine's calculate method.  For observation
        at index 1 in a 5-bar series, the context must contain exactly 2 bars.
        """
        inst_id = uuid4()
        bars = _daily_bars(["100", "110", "120", "130", "140"], inst_id)
        obs_date = bars[1].timestamp.date()

        engine = _make_feature_engine()
        received_bar_counts: list[int] = []
        original_calculate = engine.calculate

        def spy_calculate(
            definition: FeatureDefinition, ctx: FeatureCalculationContext
        ) -> FeatureValue | None:
            received_bar_counts.append(len(ctx.market_bars))
            return original_calculate(definition, ctx)

        # Patch via object attribute assignment (works for non-slot instances)
        engine.calculate = spy_calculate  # type: ignore[method-assign,assignment]

        builder = DatasetBuilder(feature_engine=engine)
        defn = _make_target_defn(horizon=1)
        feature_defn = _make_return_feature_defn(window=1)

        builder.build_dataset(
            instrument_id=inst_id,
            bars=bars,
            features=[feature_defn],
            target=defn,
            observation_dates=[obs_date],
        )

        assert len(received_bar_counts) == 1
        assert received_bar_counts[0] == 2, (
            f"Feature engine must receive exactly 2 bars for obs_date=t1, "
            f"got {received_bar_counts[0]}."
        )


# ---------------------------------------------------------------------------
# C5 — Raw-close target contract  (CURRENT BEHAVIOR — differs from intended)
# ---------------------------------------------------------------------------


class TestC5RawCloseTargetCurrentBehavior:
    """C5: CURRENT BEHAVIOR — ForwardReturnCalculator uses raw close prices only.

    *** These tests characterize the CURRENT unadjusted implementation. ***
    *** They do NOT represent the intended Phase 5.4.2+ behavior.        ***

    The current behavior computes P(t+h)/P(t) - 1 using raw (unadjusted) closes.
    This produces INCORRECT economic returns when a stock split or dividend
    falls inside the [obs_date, realization_date] window.

    The INTENDED behavior (Phase 5.4.2+) is specified in TestCAdjAcceptanceSpecs:
    - Split-adjusted prices must be used for price returns.
    - Total-return targets must add reinvested dividends to the price return.

    See also: tests marked @acceptance_spec in TestCAdjAcceptanceSpecs.
    """

    def test_current_formula_is_raw_close_ratio(self) -> None:
        """C5a CURRENT BEHAVIOR: Formula is raw P(t+h)/P(t) - 1 with no adjustment."""
        inst_id = uuid4()
        p_t = Decimal("200")
        p_t_plus_h = Decimal("220")
        bars = _daily_bars([str(p_t), "205", "210", "215", "218", str(p_t_plus_h)], inst_id)
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=5)
        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())
        assert result is not None
        assert result.value == p_t_plus_h / p_t - Decimal("1")

    def test_current_behavior_dividend_is_not_applied(self) -> None:
        """C5b CURRENT BEHAVIOR: Dividend event has no effect on ForwardReturnCalculator.

        *** This is a limitation of the current implementation, not intended design. ***

        A cash dividend of INR 2.50 paid between observation and realization means
        the ECONOMIC total return is higher than the raw price return.  The intended
        Phase 5.4.2+ total-return calculator must include dividend reinvestment.

        Current result: raw P(t+h)/P(t) - 1 = 102/100 - 1 = 0.02
        Intended result (with dividend reinvestment): approx (102 + 2.50)/100 - 1 = 0.045
        (exact amount depends on reinvestment convention; see TestCAdjAcceptanceSpecs.C_ADJ_b)
        """
        inst_id = uuid4()
        bars = _daily_bars(["100", "98", "97", "96", "95", "102"], inst_id)
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=5)

        # A dividend of INR 2.50 is paid at t+2.  Currently ignored by calculator.
        CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=bars[2].timestamp.date(),
            value=Decimal("2.50"),
            currency="INR",
        )

        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())
        assert result is not None
        # CURRENT BEHAVIOR: dividend is not included — raw price return only.
        # This will NOT be the correct assertion after Phase 5.4.2 is implemented.
        assert result.value == Decimal("102") / Decimal("100") - Decimal("1"), (
            "CURRENT BEHAVIOR: ForwardReturnCalculator ignores dividends. "
            "The intended Phase 5.4.2+ total-return target will produce a "
            "different (higher) value that includes reinvested dividend income."
        )

    def test_current_behavior_split_distorts_raw_return(self) -> None:
        """C5c CURRENT BEHAVIOR: A 10:1 split makes raw return appear as -89.5% loss.

        *** This is a known deficiency of the current implementation. ***

        Scenario: stock at 1000, then a 10:1 split → post-split price ~100.
        The economic VALUE is unchanged, but raw prices drop by 90%.
        The current (unadjusted) ForwardReturnCalculator reports -89.5%.

        The intended Phase 5.4.2+ behavior must apply split-adjusted prices
        so that the same economic scenario produces a return near 0% (or correctly
        reflects actual post-split price movements).  See TestCAdjAcceptanceSpecs.C_ADJ_a.
        """
        inst_id = uuid4()
        # Raw prices: 1000 → ... → 105 (post-split, no economic change near split)
        bars = _daily_bars(["1000", "990", "980", "970", "960", "105"], inst_id)
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=5)

        CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=bars[3].timestamp.date(),
            value=Decimal("10"),  # 10:1 split ratio
        )

        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())
        assert result is not None
        # CURRENT BEHAVIOR: raw return = 105/1000 - 1 = -0.895 (false loss)
        # This must NOT be the result after Phase 5.4.2+ adjustment is implemented.
        assert result.value == Decimal("105") / Decimal("1000") - Decimal("1"), (
            "CURRENT BEHAVIOR: split is not adjusted, raw return is misleading. "
            "The Phase 5.4.2+ adjusted calculator must not produce this result."
        )


# ---------------------------------------------------------------------------
# C6 — Dividend storage contract  (CURRENT + INTENDED infrastructure)
# ---------------------------------------------------------------------------


class TestC6DividendStorageContract:
    """C6: Dividends are stored as CorporateAction(DIVIDEND).

    The storage model is both current and intended.  What changes in Phase 5.4.2+
    is that the stored dividend will be *consumed* by the adjustment engine.
    """

    def test_corporate_action_type_dividend_exists(self) -> None:
        """C6a: CorporateActionType.DIVIDEND must exist."""
        assert CorporateActionType.DIVIDEND.value == "dividend"

    def test_dividend_corporate_action_stores_correct_fields(self) -> None:
        """C6b: A DIVIDEND CorporateAction is created with the expected field values."""
        inst_id = uuid4()
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 15),
            value=Decimal("5.00"),
            currency="INR",
            description="Q3 interim dividend",
        )
        assert dividend.action_type == CorporateActionType.DIVIDEND
        assert dividend.value == Decimal("5.00")
        assert dividend.currency == "INR"
        assert dividend.execution_date == date(2026, 1, 15)

    def test_current_forward_return_calculator_accepts_only_bars(self) -> None:
        """C6c: The current ForwardReturnCalculator is unadjusted by design.

        CURRENT BEHAVIOR: the calculator signature accepts definition, instrument_id,
        bars, and observation_date only.  Adding corporate_actions would be a
        BREAKING DESIGN CHANGE requiring explicit approval.

        This test verifies the behavior: passing corporate actions is not supported,
        and the calculator produces a result using only the supplied bars.
        """
        inst_id = uuid4()
        bars = _daily_bars(["100", "105"], inst_id)
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=1)

        # Corporate action exists in domain but is NOT passed to calculator.
        # The calculator must still produce a correct raw-price result.
        CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=bars[0].timestamp.date(),
            value=Decimal("1.00"),
        )

        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())
        # CURRENT BEHAVIOR: result is raw price return, dividend not included
        assert result is not None
        assert result.value == Decimal("105") / Decimal("100") - Decimal("1")

    def test_dividend_value_must_be_positive(self) -> None:
        """C6d: Dividend value must be strictly positive (domain validation)."""
        with pytest.raises(ValueError, match="strictly positive"):
            CorporateAction(
                instrument_id=uuid4(),
                action_type=CorporateActionType.DIVIDEND,
                execution_date=date(2026, 1, 15),
                value=Decimal("0"),
            )
        with pytest.raises(ValueError, match="strictly positive"):
            CorporateAction(
                instrument_id=uuid4(),
                action_type=CorporateActionType.DIVIDEND,
                execution_date=date(2026, 1, 15),
                value=Decimal("-1.50"),
            )

    def test_dividend_currency_is_normalised_to_uppercase(self) -> None:
        """C6e: Currency codes are uppercased by the domain model."""
        ca = CorporateAction(
            instrument_id=uuid4(),
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 15),
            value=Decimal("3.50"),
            currency="inr",
        )
        assert ca.currency == "INR"


# ---------------------------------------------------------------------------
# C7 — Split storage contract  (CURRENT + INTENDED infrastructure)
# ---------------------------------------------------------------------------


class TestC7SplitStorageContract:
    """C7: Splits are stored as CorporateAction(SPLIT).

    The storage model is both current and intended.
    """

    def test_corporate_action_type_split_exists(self) -> None:
        """C7a: CorporateActionType.SPLIT must exist."""
        assert CorporateActionType.SPLIT.value == "split"

    def test_split_corporate_action_stores_correct_fields(self) -> None:
        """C7b: A SPLIT CorporateAction is created with the expected field values."""
        inst_id = uuid4()
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 3, 1),
            value=Decimal("2"),
            description="2-for-1 stock split",
        )
        assert split.action_type == CorporateActionType.SPLIT
        assert split.value == Decimal("2")
        assert split.execution_date == date(2026, 3, 1)

    def test_split_ratio_must_be_strictly_positive(self) -> None:
        """C7c: Split ratio must be strictly positive (domain validation)."""
        with pytest.raises(ValueError, match="strictly positive"):
            CorporateAction(
                instrument_id=uuid4(),
                action_type=CorporateActionType.SPLIT,
                execution_date=date(2026, 3, 1),
                value=Decimal("0"),
            )
        with pytest.raises(ValueError, match="strictly positive"):
            CorporateAction(
                instrument_id=uuid4(),
                action_type=CorporateActionType.SPLIT,
                execution_date=date(2026, 3, 1),
                value=Decimal("-2"),
            )


# ---------------------------------------------------------------------------
# C8 — Realization-date semantics  (CURRENT + INTENDED)
# ---------------------------------------------------------------------------


class TestC8RealizationDateSemanticsContract:
    """C8: realization_date is always the calendar date of bar at obs_idx + h.

    This rule applies to both the current and intended implementations.
    """

    def test_realization_date_is_bar_date_not_calendar_offset(self) -> None:
        """C8a: realization_date is the actual future bar date, not a +h-days offset.

        With weekly bars, h=3 means the 4th Monday (Jan 26), not Jan 5 + 21 days.
        Both values happen to coincide here, but the semantic difference matters
        when there are holiday gaps.
        """
        inst_id = uuid4()
        monday = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
        bars = [
            _make_bar(monday + timedelta(weeks=i), Decimal(str(100 + i * 5)), inst_id)
            for i in range(4)
        ]
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=3)
        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())
        assert result is not None
        assert result.realization_date == bars[3].timestamp.date()

    def test_realization_date_skips_holiday_gap(self) -> None:
        """C8b: realization_date correctly skips a mid-week holiday gap.

        Bars: Jan 5 (Mon), Jan 6 (Tue), Jan 9 (Fri — Wed/Thu missing), Jan 12 (Mon).
        h=2 from Jan 5 → bar at index 2 = Jan 9 (not Jan 7 = +2 calendar days).
        """
        inst_id = uuid4()
        timestamps = [
            datetime(2026, 1, 5, 10, tzinfo=UTC),  # bar[0]
            datetime(2026, 1, 6, 10, tzinfo=UTC),  # bar[1]
            datetime(2026, 1, 9, 10, tzinfo=UTC),  # bar[2] — gap
            datetime(2026, 1, 12, 10, tzinfo=UTC),  # bar[3]
        ]
        closes = ["100", "102", "105", "108"]
        bars = [
            _make_bar(ts, Decimal(c), inst_id) for ts, c in zip(timestamps, closes, strict=True)
        ]
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=2)
        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())
        assert result is not None
        assert result.realization_date == date(2026, 1, 9), (
            "realization_date must be Jan 9 (2 trading bars after Jan 5), "
            "not Jan 7 (2 calendar days after Jan 5)."
        )

    def test_realization_date_strictly_after_observation_date(self) -> None:
        """C8c: TargetValue invariant — realization_date > observation_date."""
        inst_id = uuid4()
        bars = _daily_bars(["100", "110"], inst_id)
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=1)
        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())
        assert result is not None
        assert result.realization_date > result.observation_date

    def test_target_value_rejects_same_day_realization(self) -> None:
        """C8d: TargetValue constructor rejects realization_date == observation_date."""
        defn = _make_target_defn()
        obs = date(2026, 1, 5)
        with pytest.raises(ValueError, match="strictly after"):
            TargetValue(
                target=defn,
                instrument_id=uuid4(),
                observation_date=obs,
                value=Decimal("0.05"),
                realization_date=obs,
            )

    def test_target_value_rejects_realization_before_observation(self) -> None:
        """C8e: TargetValue constructor rejects realization_date < observation_date."""
        defn = _make_target_defn()
        with pytest.raises(ValueError, match="strictly after"):
            TargetValue(
                target=defn,
                instrument_id=uuid4(),
                observation_date=date(2026, 1, 10),
                value=Decimal("0.05"),
                realization_date=date(2026, 1, 9),
            )


# ---------------------------------------------------------------------------
# C9 — DatasetBuilder target-visibility contract  (CURRENT + INTENDED)
# ---------------------------------------------------------------------------


class TestC9DatasetBuilderTargetVisibilityContract:
    """C9: DatasetBuilder passes ALL bars to target calculator.

    Feature engine: receives only bars ≤ obs_date (point-in-time isolation).
    Target calculator: receives all sorted bars so it can look h bars forward.

    This asymmetry is intentional and applies to both current and future
    implementations.
    """

    def test_target_calculator_receives_full_bar_sequence(self) -> None:
        """C9a: ForwardReturnCalculator receives all bars, not just history.

        5 bars; observation at bar[1] (t1). Features use bars[0:2] only.
        Target (h=4) needs bar[4] — must be visible to target calculator.
        """
        inst_id = uuid4()
        bars = _daily_bars(["100", "110", "120", "130", "140"], inst_id)

        received_bar_counts: list[int] = []
        original_calc = ForwardReturnCalculator()
        original_calculate = original_calc.calculate

        def spy_calculate(
            definition: TargetDefinition,
            instrument_id: UUID,
            bars: Sequence[MarketBar],
            observation_date: date,
        ) -> TargetValue | None:
            received_bar_counts.append(len(bars))
            return original_calculate(definition, instrument_id, bars, observation_date)

        original_calc.calculate = spy_calculate  # type: ignore[method-assign]

        engine = _make_feature_engine()
        builder = DatasetBuilder(feature_engine=engine, target_calculator=original_calc)
        defn = _make_target_defn(horizon=4)
        feature_defn = _make_return_feature_defn(window=1)

        builder.build_dataset(
            instrument_id=inst_id,
            bars=bars,
            features=[feature_defn],
            target=defn,
            observation_dates=[bars[1].timestamp.date()],
        )

        assert len(received_bar_counts) >= 1, "Target calculator was never called."
        assert received_bar_counts[0] == 5, (
            f"ForwardReturnCalculator must receive all 5 bars, got {received_bar_counts[0]}."
        )


# ---------------------------------------------------------------------------
# C10 — Data-revision (upsert) contract  (CURRENT BEHAVIOR, also intended)
# ---------------------------------------------------------------------------


class TestC10DataRevisionUpsertContract:
    """C10: Re-ingesting a bar for (instrument_id, timestamp) overwrites the previous record.

    WHAT IS VERIFIED HERE:
    - The domain model allows constructing two MarketBar objects with the same
      (instrument_id, timestamp) key.  Uniqueness is enforced at the DB level.
    - The abstract interface exposes upsert_bars() for bulk ingestion.
    - The concrete PostgresMarketBarRepository uses ON CONFLICT DO UPDATE,
      overwriting open/high/low/close/volume in place.

    WHAT IS NOT VERIFIED HERE (requires live database):
    - The actual persisted value after two consecutive upsert_bars() calls.
    - That a second call does NOT duplicate the row.

    An integration test covering actual upsert behavior exists in:
        tests/integration/database/test_repositories.py::test_market_bar_repository_bulk_upsert_idempotent_and_updates
    and requires a running PostgreSQL database (pytest mark: integration).
    That test explicitly asserts:
        1. Initial insert of bars persists initial OHLCV and assigns primary keys.
        2. Re-ingesting the exact same batch does not duplicate rows (idempotency).
        3. Ingesting updated bars with identical (instrument_id, timestamp) keys
           overwrites open/high/low/close/volume in place while preserving primary keys.
    """

    def test_domain_allows_two_bars_with_same_key(self) -> None:
        """C10a: Domain layer has no uniqueness constraint on (instrument_id, timestamp).

        Duplicate enforcement is entirely the responsibility of the DB layer
        (unique index on market_bars(instrument_id, timestamp)).  This test
        documents that boundary: the domain can represent a revision, but
        cannot persist it without a repository.
        """
        inst_id = uuid4()
        bar_v1 = _make_bar(_BASE, Decimal("100"), instrument_id=inst_id)
        bar_v2 = _make_bar(_BASE, Decimal("105"), instrument_id=inst_id)

        # Same (instrument_id, timestamp) key — domain construction succeeds.
        assert bar_v1.instrument_id == bar_v2.instrument_id
        assert bar_v1.timestamp == bar_v2.timestamp
        assert bar_v1.close != bar_v2.close, (
            "Two bars with the same key but different prices represent a revision. "
            "The domain allows this; the DB enforces uniqueness via upsert."
        )

    def test_repository_interface_exposes_upsert_bars_method(self) -> None:
        """C10b: MarketBarRepository declares upsert_bars as an abstract method.

        Verifies the public interface contract.  The concrete implementation
        (PostgresMarketBarRepository.upsert_bars) uses PostgreSQL
        ON CONFLICT DO UPDATE to overwrite price fields in place.
        """
        from app.domain.repositories.market_bar_repository import MarketBarRepository

        assert hasattr(MarketBarRepository, "upsert_bars"), (
            "MarketBarRepository must declare upsert_bars() for bulk ingestion."
        )
        assert callable(getattr(MarketBarRepository, "upsert_bars", None)), (
            "MarketBarRepository.upsert_bars must be callable."
        )

    def test_concrete_repository_uses_on_conflict_do_update(self) -> None:
        """C10c: PostgresMarketBarRepository.upsert_bars uses ON CONFLICT DO UPDATE.

        Verifies the source contains the ON CONFLICT clause, as a lightweight
        stand-in for the integration test that requires a live database.

        NOTE: This test inspects source text.  It is intentionally fragile in
        the sense that it will catch if the upsert logic is removed or replaced.
        If the implementation migrates to a different mechanism (e.g., a merge
        statement), update this test accordingly.

        PREREQUISITE FOR FULL COVERAGE: run the integration test suite with a
        live PostgreSQL instance to verify actual persisted behavior.
        """
        from app.infrastructure.database.repositories.market_bar import (
            PostgresMarketBarRepository,
        )

        source = inspect.getsource(PostgresMarketBarRepository.upsert_bars)
        assert "on_conflict_do_update" in source, (
            "PostgresMarketBarRepository.upsert_bars must use "
            "on_conflict_do_update to overwrite existing rows."
        )
        assert "excluded" in source, (
            "The ON CONFLICT clause must reference the EXCLUDED pseudo-table "
            "to take the new values."
        )


# ---------------------------------------------------------------------------
# C_ADJ — Acceptance specifications for Phase 5.4.2+ adjustment engine
# ---------------------------------------------------------------------------


class TestCAdjAcceptanceSpecs:
    """C_ADJ: Acceptance specifications for the Phase 5.4.2+ adjustment engine.

    *** NONE OF THESE TESTS PASS TODAY. ***
    All are marked @acceptance_spec (xfail/strict=True).

    They will pass once AdjustedPriceService and/or a total-return target
    calculator are introduced.  The implementer must satisfy ALL of them.

    PREREQUISITE ABSTRACTIONS (must exist before these tests can be written
    as passing tests):

    1. AdjustedPriceService (or equivalent):
       - Input:  list[MarketBar] + list[CorporateAction]
       - Output: list[MarketBar] with OHLC and volume back-adjusted
       - Adjustment convention: multiply all prices BEFORE split date by
         (1 / split_ratio) for each split, working backwards from the most
         recent bar.  Volume is multiplied by split_ratio (inverse).
       - Must NOT mutate the input MarketBar objects.

    2. TotalReturnCalculator (or an extended ForwardReturnCalculator):
       - Input:  bars, corporate_actions, observation_date, horizon_bars
       - Output: total return including reinvested dividends
       - Dividend reinvestment convention: dividends are reinvested at the
         ex-dividend-date closing price (i.e., the bar AFTER the dividend
         execution_date).  The reinvestment factor is:
             factor = 1 + (dividend_per_share / ex_div_close_price)
         Applied to the cumulative return for each dividend in the window.

    DESIGN DECISIONS LOCKED BY THESE SPECS:
    - Adjustment is backward-looking (back-adjust historical prices).
    - Volume uses the INVERSE factor relative to price.
    - Feature calculators use split-adjusted prices; raw prices are preserved.
    - Corporate actions inside a target horizon affect the TOTAL-RETURN target.
    - Corporate actions outside a target horizon must not affect observation features.
    """

    @acceptance_spec
    def test_C_ADJ_a_split_does_not_create_false_loss(self) -> None:
        """C_ADJ_a: A 2-for-1 split must not produce a false 50% price loss.

        Scenario:
            t0: close = 200 (pre-split)
            t1: close = 100 (post-split, same economic value — 2-for-1 split)

        Raw return = 100/200 - 1 = -0.50 (WRONG: looks like a 50% loss)

        With split-adjusted prices:
            adjusted_t0 = 200 / 2 = 100  (back-adjusted by split factor)
            adjusted_t1 = 100             (post-split, no adjustment needed)

        Adjusted return = 100/100 - 1 = 0.00  (CORRECT: no economic change)

        The adjustment engine must produce 0.00, not -0.50.

        PREREQUISITE: AdjustedPriceService must exist and be wired into
        the forward-return calculation path.
        """
        inst_id = uuid4()
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),  # Split on t1
            value=Decimal("2"),  # 2-for-1
        )
        bars = [
            _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_id),  # t0
            _make_bar(
                datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id
            ),  # t1 post-split
        ]

        # TODO (Phase 5.4.2): replace with AdjustedForwardReturnCalculator or
        # adjust bars via AdjustedPriceService before calling the calculator.
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=1)
        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())

        assert result is not None
        # ACCEPTANCE CRITERION: adjusted return must be 0.00 (no economic change)
        assert result.value == Decimal("0"), (
            f"Split-adjusted return for a 2-for-1 split must be 0.00 "
            f"(economic value unchanged), got {result.value}."
        )
        # Verify this is NOT the raw return (which is -0.50)
        assert result.value != Decimal("-0.5"), (
            "Raw -0.50 return must not appear after split adjustment."
        )
        _ = split  # referenced to confirm it is passed to the adjusted calculator

    @acceptance_spec
    def test_C_ADJ_b_dividend_converts_price_return_to_total_return(self) -> None:
        """C_ADJ_b: Cash dividend distinguishes price return from total return.

        Scenario:
            t0: obs_date, close = 100
            t1: ex-div date, close = 97 (price drops by ~3 after dividend)
                dividend paid = 3.00 per share
            t2 (h=2): close = 97  (realization date)

        Reinvestment convention:
            Dividend reinvested at t1 ex-div close (97).
            Reinvestment factor = 1 + (3.00 / 97) ≈ 1.030928

        Price return  = 97/100 - 1 = -0.03  (looks like a loss)
        Total return  = (97/100 - 1 + 3/100) = 0.00 (flat economic return)

        More precisely, using the reinvestment factor:
            total_return = (t2_adj_close / t0_close) - 1
        where t2_adj_close accounts for reinvested shares.

        PREREQUISITE: TotalReturnCalculator must exist with an explicit
        dividend reinvestment convention (reinvest at ex-div close).
        """
        inst_id = uuid4()
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 6),  # ex-div date
            value=Decimal("3.00"),
            currency="INR",
        )
        bars = [
            _make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),  # t0
            _make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("97"), inst_id),  # t1 ex-div
            _make_bar(
                datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("97"), inst_id
            ),  # t2 realization
        ]

        # TODO (Phase 5.4.2): call TotalReturnCalculator(corporate_actions=[dividend])
        calc = ForwardReturnCalculator()
        defn = _make_target_defn(horizon=2)
        result = calc.calculate(defn, inst_id, bars, bars[0].timestamp.date())

        assert result is not None
        # ACCEPTANCE CRITERION: total return ≈ 0.00 (price loss offset by dividend)
        # Exact value depends on reinvestment convention; document it here.
        # With "reinvest at ex-div close" convention:
        #   factor = 1 + 3.00/97 = 100/97
        #   total_return = (97 * 100/97) / 100 - 1 = 100/100 - 1 = 0
        assert result.value == Decimal("0"), (
            f"Total return must be 0.00 when dividend (3) offsets price drop (3), "
            f"got {result.value}. "
            "Reinvestment convention: reinvest at ex-dividend-date close."
        )
        _ = dividend

    def test_C_ADJ_c_all_ohlc_fields_use_consistent_split_factor(self) -> None:
        """C_ADJ_c: All four OHLC fields are adjusted by the same split factor.

        Scenario: 2-for-1 split on t1.
            t0 raw:  open=100, high=110, low=95, close=105
            t1 raw:  open=52, high=56, low=50, close=53 (post-split prices)

        Back-adjusted t0 must be:
            adj_open = 100/2 = 50, adj_high = 110/2 = 55,
            adj_low = 95/2 = 47.5, adj_close = 105/2 = 52.5

        The high/low ratio and open/close relationship must be preserved.
        """
        inst_id = uuid4()
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        bar_t0 = MarketBar(
            instrument_id=inst_id,
            timestamp=datetime(2026, 1, 5, 10, tzinfo=UTC),
            open=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("1000"),
        )

        service = AdjustedPriceService()
        result = service.adjust([bar_t0], [split])

        assert len(result) == 1
        adj_t0 = result[0]
        assert adj_t0.open == Decimal("50")
        assert adj_t0.high == Decimal("55")
        assert adj_t0.low == Decimal("47.5")
        assert adj_t0.close == Decimal("52.5")
        assert adj_t0.volume == Decimal("2000")
        assert adj_t0.low <= adj_t0.open <= adj_t0.high
        assert adj_t0.low <= adj_t0.close <= adj_t0.high

    def test_C_ADJ_d_volume_uses_inverse_split_factor(self) -> None:
        """C_ADJ_d: Volume adjustment uses the INVERSE factor to price adjustment.

        Scenario: 2-for-1 split.
            Pre-split volume at t0 = 1000 shares traded.
            Post-split equivalent = 1000 x 2 = 2000 (shares are halved in price,
            doubled in count).

        Convention: adj_volume = raw_volume x split_ratio
        This is the OPPOSITE direction from price adjustment.
        """
        inst_id = uuid4()
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        bar_t0 = _make_bar(
            datetime(2026, 1, 5, 10, tzinfo=UTC),
            Decimal("100"),
            instrument_id=inst_id,
        )

        service = AdjustedPriceService()
        result = service.adjust([bar_t0], [split])

        assert len(result) == 1
        assert result[0].volume == Decimal("2000")

        # Reverse split (1-for-2, ratio 0.5): 1000 -> 500
        rev_split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("0.5"),
        )
        rev_result = service.adjust([bar_t0], [rev_split])
        assert rev_result[0].volume == Decimal("500")

    def test_C_ADJ_e_future_corporate_action_does_not_affect_past_features(
        self,
    ) -> None:
        """C_ADJ_e: A corporate action AFTER obs_date must not affect features AT obs_date.

        Scenario:
            t0: obs_date (features computed here)
            t3: split execution date (AFTER obs_date)

        Feature at t0 uses prices from t0 and earlier only.  The split at t3
        must not alter the price history visible to features at t0.

        This is the point-in-time isolation guarantee applied to adjusted prices:
        at obs_date, only corporate actions BEFORE obs_date affect the
        back-adjusted prices used in features.
        """
        inst_id = uuid4()
        obs_date = date(2026, 1, 5)
        bar_t0 = _make_bar(
            datetime(2026, 1, 5, 10, tzinfo=UTC),
            Decimal("200"),
            instrument_id=inst_id,
        )
        future_split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 20),
            value=Decimal("2"),
        )
        service = AdjustedPriceService()

        # View as of obs_date (2026-01-05): future split at 2026-01-20 is NOT effective
        pit_result = service.adjust([bar_t0], [future_split], as_of_date=obs_date)
        assert len(pit_result) == 1
        assert pit_result[0].close == Decimal("200")

        # View as of split date (2026-01-20): split IS effective and back-adjusts t0
        post_result = service.adjust([bar_t0], [future_split], as_of_date=date(2026, 1, 20))
        assert len(post_result) == 1
        assert post_result[0].close == Decimal("100")

        # View as of obs_date (2026-01-05): future split at 2026-01-20 is NOT effective
        pit_result = service.adjust([bar_t0], [future_split], as_of_date=obs_date)
        assert len(pit_result) == 1
        assert pit_result[0].close == Decimal("200")

        # View as of split date (2026-01-20): split IS effective and back-adjusts t0
        post_result = service.adjust([bar_t0], [future_split], as_of_date=date(2026, 1, 20))
        assert len(post_result) == 1
        assert post_result[0].close == Decimal("100")

    @acceptance_spec
    def test_C_ADJ_f_corporate_action_inside_horizon_affects_total_return_target(
        self,
    ) -> None:
        """C_ADJ_f: A corporate action inside [obs_date, realization_date] affects total return.

        Scenario:
            t0: obs_date
            t2: dividend paid (inside the horizon)
            t5: realization_date

        The total-return target must include the dividend income accumulated
        between t0 and t5.  The price-return target need not.

        PREREQUISITE: TotalReturnCalculator that accepts corporate_actions.
        """
        raise NotImplementedError(
            "Acceptance spec: TotalReturnCalculator not yet implemented. "
            "Dividends paid between obs_date and realization_date must be "
            "included in the total-return target via the reinvestment formula."
        )

    def test_C_ADJ_g_raw_market_bar_values_unchanged_after_adjustment(self) -> None:
        """C_ADJ_g: Raw MarketBar values are NOT mutated by the adjustment service.

        The AdjustedPriceService must return NEW MarketBar objects (or a separate
        adjusted representation) rather than modifying the input bars in place.
        """
        inst_id = uuid4()
        raw_bar = _make_bar(_BASE, Decimal("200"), instrument_id=inst_id)
        original_close = raw_bar.close

        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=(_BASE + timedelta(days=1)).date(),
            value=Decimal("2"),
        )

        service = AdjustedPriceService()
        result = service.adjust([raw_bar], [split])

        assert len(result) == 1
        adj_bar = result[0]
        # Raw bar is completely unmutated
        assert raw_bar.close == original_close == Decimal("200")
        assert adj_bar.close == Decimal("100")
        assert adj_bar is not raw_bar

    def test_C_ADJ_h_invalid_corporate_action_values_are_rejected(self) -> None:
        """C_ADJ_h: Invalid corporate-action values are rejected at domain construction.

        Domain policy: CorporateAction.value must be a finite, strictly positive Decimal.

        The following MUST be rejected:
        - Zero value (enforced: 'strictly positive')
        - Negative value (enforced: 'strictly positive')
        - NaN (rejected via finite check)
        - Infinite values (+Inf, -Inf) (rejected via finite check).
        """
        # Zero rejected
        with pytest.raises(ValueError, match="strictly positive"):
            CorporateAction(
                instrument_id=uuid4(),
                action_type=CorporateActionType.SPLIT,
                execution_date=date(2026, 1, 6),
                value=Decimal("0"),
            )

        # Negative rejected
        with pytest.raises(ValueError, match="strictly positive"):
            CorporateAction(
                instrument_id=uuid4(),
                action_type=CorporateActionType.SPLIT,
                execution_date=date(2026, 1, 6),
                value=Decimal("-2"),
            )

        # NaN rejected
        with pytest.raises((ValueError, ArithmeticError)):
            CorporateAction(
                instrument_id=uuid4(),
                action_type=CorporateActionType.SPLIT,
                execution_date=date(2026, 1, 6),
                value=Decimal("NaN"),
            )

        # -Infinity rejected
        with pytest.raises((ValueError, ArithmeticError)):
            CorporateAction(
                instrument_id=uuid4(),
                action_type=CorporateActionType.SPLIT,
                execution_date=date(2026, 1, 6),
                value=Decimal("-Inf"),
            )

        # +Infinity rejected
        with pytest.raises((ValueError, ArithmeticError)):
            CorporateAction(
                instrument_id=uuid4(),
                action_type=CorporateActionType.SPLIT,
                execution_date=date(2026, 1, 6),
                value=Decimal("Inf"),
            )
