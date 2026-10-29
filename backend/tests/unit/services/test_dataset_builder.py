from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from uuid import UUID, uuid4

import pytest

from app.domain.entities.corporate_action import CorporateAction
from app.domain.entities.market_bar import MarketBar
from app.domain.enums.corporate_action_type import CorporateActionType
from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.prediction.dataset import DatasetRow, FeatureMatrix
from app.domain.prediction.target_definition import TargetDefinition
from app.domain.repositories.corporate_action_repository import (
    CorporateActionRepository,
)
from app.services.features.feature_calculator_registry import create_default_feature_registry
from app.services.features.feature_engine import FeatureEngine
from app.services.prediction.dataset_builder import DatasetBuilder, generate_feature_name
from app.services.prediction.dataset_splitter import DatasetSplitter, SplitConfig

# ===========================================================================
# Helpers & Fixtures
# ===========================================================================


def make_bar(
    timestamp: datetime,
    close: Decimal,
    instrument_id: UUID,
    high: Decimal | None = None,
    low: Decimal | None = None,
    volume: Decimal = Decimal("5000"),
) -> MarketBar:
    bar_close = close
    bar_high = high if high is not None else bar_close
    bar_low = low if low is not None else bar_close
    return MarketBar(
        instrument_id=instrument_id,
        timestamp=timestamp,
        open=bar_close,
        high=bar_high,
        low=bar_low,
        close=bar_close,
        volume=volume,
    )


def make_daily_series(
    start_date: date,
    closes: list[str],
    instrument_id: UUID,
) -> list[MarketBar]:
    bars: list[MarketBar] = []
    for i, c_str in enumerate(closes):
        d = start_date + timedelta(days=i)
        ts = datetime(d.year, d.month, d.day, 10, 0, tzinfo=UTC)
        c = Decimal(c_str)
        bars.append(
            make_bar(
                timestamp=ts,
                close=c,
                instrument_id=instrument_id,
                high=c + Decimal("2"),
                low=c - Decimal("2"),
            )
        )
    return bars


@pytest.fixture
def engine() -> FeatureEngine:
    registry = create_default_feature_registry()
    return FeatureEngine(registry)


@pytest.fixture
def builder(engine: FeatureEngine) -> DatasetBuilder:
    return DatasetBuilder(feature_engine=engine)


# ===========================================================================
# Unit Tests
# ===========================================================================


class TestDatasetBuilderCore:
    def test_end_to_end_dataset_construction(self, builder: DatasetBuilder) -> None:
        """1. End-to-end dataset construction with multiple features and a 5-bar target."""
        inst_id = uuid4()
        # 30 daily bars with linearly increasing prices
        closes = [str(100 + i) for i in range(30)]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d return",
                parameters={"window": 1},
            ),
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="5d SMA",
                parameters={"window": 5},
            ),
            FeatureDefinition(
                name="rsi",
                category=FeatureCategory.TECHNICAL,
                description="14d RSI",
                parameters={"window": 14},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=5)

        matrix = builder.build_dataset(
            instrument_id=inst_id,
            bars=bars,
            features=features,
            target=target,
        )

        assert isinstance(matrix, FeatureMatrix)
        assert matrix.instrument_id == inst_id
        assert matrix.target_definition == target
        assert len(matrix.feature_names) == 3
        assert matrix.feature_names == (
            "return_window_1",
            "sma_window_5",
            "rsi_window_14",
        )

        # Lookback requirement: RSI(14) requires 15 bars (indices 0..14).
        # Thus the first 14 bars (indices 0..13) lack complete features -> dropped.
        # Total bars = 30.
        # Horizon requirement: 5 bars. Bars at indices 25..29 lack 5 future bars -> dropped.
        # Complete rows are from index 14 to index 24 inclusive: 24 - 14 + 1 = 11 rows.
        assert len(matrix) == 11
        assert matrix.dropped_incomplete_feature_count == 14
        assert matrix.dropped_incomplete_target_count == 5

        # Check shapes of matrix properties
        assert len(matrix.feature_matrix) == 11
        assert all(len(row) == 3 for row in matrix.feature_matrix)
        assert len(matrix.target_vector) == 11
        assert len(matrix.observation_dates) == 11
        assert len(matrix.realization_dates) == 11

    def test_correct_feature_and_target_alignment_for_known_observation_date(
        self, builder: DatasetBuilder
    ) -> None:
        """2. Correct feature and target alignment for a known observation date."""
        inst_id = uuid4()
        # 10 bars: close = 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
        closes = [str((i + 1) * 10) for i in range(10)]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d return",
                parameters={"window": 1},
            ),
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="3d SMA",
                parameters={"window": 3},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=2)

        # Explicitly query observation date 2026-01-04 (index 3, close 40)
        matrix = builder.build_dataset(
            instrument_id=inst_id,
            bars=bars,
            features=features,
            target=target,
            observation_dates=[date(2026, 1, 4)],
        )

        assert len(matrix) == 1
        row = matrix.rows[0]
        assert row.observation_date == date(2026, 1, 4)
        # Horizon 2: future bar is index 3 + 2 = 5 (date 2026-01-06, close 60)
        assert row.realization_date == date(2026, 1, 6)

        # Expected features at index 3 (close 40, prev 30):
        # return_1 = (40 / 30) - 1 = 10/30 = 1/3
        expected_ret = (Decimal("40") / Decimal("30")) - Decimal("1")
        # sma_3 = (20 + 30 + 40) / 3 = 30
        expected_sma = Decimal("30")
        assert row.feature_values[0] == expected_ret
        assert row.feature_values[1] == expected_sma

        # Expected target: (60 / 40) - 1 = 0.50
        expected_target = Decimal("0.50")
        assert row.target_value == expected_target

    def test_strict_point_in_time_slicing_no_future_leakage(self, builder: DatasetBuilder) -> None:
        """3. Strict point-in-time slicing: future bars are not visible to feature engine."""
        inst_id = uuid4()
        # 10 bars
        closes = [str(100 + i) for i in range(10)]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)

        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="5d SMA",
                parameters={"window": 5},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=2)

        # Calculate dataset for date 2026-01-06 (index 5)
        obs_date = date(2026, 1, 6)
        matrix = builder.build_dataset(
            instrument_id=inst_id,
            bars=bars,
            features=features,
            target=target,
            observation_dates=[obs_date],
        )

        assert len(matrix) == 1
        row = matrix.rows[0]
        # At index 5: closes are 100, 101, 102, 103, 104, 105.
        # Trailing 5 closes are 101, 102, 103, 104, 105 -> mean = 103.0
        assert row.feature_values[0] == Decimal("103")

    def test_changing_future_prices_leaves_features_unchanged_while_target_changes(
        self, builder: DatasetBuilder
    ) -> None:
        """4. Modifying prices after date t does NOT change features at t, but updates target."""
        inst_id = uuid4()
        closes_1 = ["100", "102", "104", "106", "108", "110", "112"]
        bars_1 = make_daily_series(date(2026, 1, 1), closes_1, inst_id)

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d return",
                parameters={"window": 1},
            ),
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="3d SMA",
                parameters={"window": 3},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=2)

        obs_date = date(2026, 1, 4)  # index 3, close 106
        matrix_1 = builder.build_dataset(
            instrument_id=inst_id,
            bars=bars_1,
            features=features,
            target=target,
            observation_dates=[obs_date],
        )

        # Now create bars_2 with wild prices after date 2026-01-04 (indices 4, 5, 6)
        closes_2 = ["100", "102", "104", "106", "999999", "888888", "777777"]
        bars_2 = make_daily_series(date(2026, 1, 1), closes_2, inst_id)

        matrix_2 = builder.build_dataset(
            instrument_id=inst_id,
            bars=bars_2,
            features=features,
            target=target,
            observation_dates=[obs_date],
        )

        # Features at date t must be IDENTICAL
        assert matrix_1.rows[0].feature_values == matrix_2.rows[0].feature_values

        # Targets must be DIFFERENT because future price at index 5 changed from 110 to 888888
        assert matrix_1.rows[0].target_value != matrix_2.rows[0].target_value
        expected_target_2 = (Decimal("888888") / Decimal("106")) - Decimal("1")
        assert matrix_2.rows[0].target_value == expected_target_2

    def test_deterministic_column_ordering_preserves_input_sequence(
        self, builder: DatasetBuilder
    ) -> None:
        """5. Column order strictly preserves the sequence supplied in features."""
        inst_id = uuid4()
        closes = [str(100 + i) for i in range(25)]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)
        target = TargetDefinition(name="forward_return", horizon_bars=2)

        def_ret = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="1d",
            parameters={"window": 1},
        )
        def_sma = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="5d",
            parameters={"window": 5},
        )
        def_rsi = FeatureDefinition(
            name="rsi",
            category=FeatureCategory.TECHNICAL,
            description="14d",
            parameters={"window": 14},
        )

        # Order A: [ret, sma, rsi]
        matrix_a = builder.build_dataset(inst_id, bars, [def_ret, def_sma, def_rsi], target)
        assert matrix_a.feature_names == (
            "return_window_1",
            "sma_window_5",
            "rsi_window_14",
        )

        # Order B: [rsi, ret, sma]
        matrix_b = builder.build_dataset(inst_id, bars, [def_rsi, def_ret, def_sma], target)
        assert matrix_b.feature_names == (
            "rsi_window_14",
            "return_window_1",
            "sma_window_5",
        )

        # Values in rows must also align with column ordering
        row_a = matrix_a.rows[0]
        row_b = matrix_b.rows[0]
        # In row_a: (ret, sma, rsi)
        # In row_b: (rsi, ret, sma)
        assert row_a.feature_values[0] == row_b.feature_values[1]  # ret
        assert row_a.feature_values[1] == row_b.feature_values[2]  # sma
        assert row_a.feature_values[2] == row_b.feature_values[0]  # rsi

    def test_distinct_names_for_different_parameterizations_of_same_family(
        self, builder: DatasetBuilder
    ) -> None:
        """6. Distinct names for different parameterizations of the same calculator."""
        inst_id = uuid4()
        closes = [str(100 + i) for i in range(30)]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)
        target = TargetDefinition(name="forward_return", horizon_bars=1)

        sma_5 = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="5d",
            parameters={"window": 5},
        )
        sma_20 = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="20d",
            parameters={"window": 20},
        )
        bb_upper = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="upper",
            parameters={"window": 20, "band": "upper", "k": 2},
        )
        bb_lower = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="lower",
            parameters={"window": 20, "band": "lower", "k": 2},
        )

        matrix = builder.build_dataset(inst_id, bars, [sma_5, sma_20, bb_upper, bb_lower], target)

        assert matrix.feature_names == (
            "sma_window_5",
            "sma_window_20",
            "bollinger_bands_band_upper_k_2_window_20",
            "bollinger_bands_band_lower_k_2_window_20",
        )
        assert len(matrix.feature_names) == 4

    def test_rejection_of_duplicate_feature_definitions_or_column_collisions(
        self, builder: DatasetBuilder
    ) -> None:
        """7. Rejection of duplicate feature configurations or generated column collisions."""
        inst_id = uuid4()
        bars = make_daily_series(date(2026, 1, 1), ["100", "101", "102"], inst_id)
        target = TargetDefinition(name="forward_return", horizon_bars=1)

        # Duplicate definition
        sma_5_a = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="5d a",
            parameters={"window": 5},
        )
        sma_5_b = FeatureDefinition(
            name="sma",
            category=FeatureCategory.TECHNICAL,
            description="5d b",
            parameters={"window": 5},
        )

        with pytest.raises(ValueError, match="Duplicate feature configuration"):
            builder.build_dataset(inst_id, bars, [sma_5_a, sma_5_b], target)

    def test_missing_feature_values_cause_row_drop_counted_once(
        self, builder: DatasetBuilder
    ) -> None:
        """8. Missing feature values cause a row to be dropped and counted once."""
        inst_id = uuid4()
        # 10 bars
        closes = [str(100 + i) for i in range(10)]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)

        # SMA 5 requires 5 bars. Bars at indices 0, 1, 2, 3 have < 5 bars.
        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="5d",
                parameters={"window": 5},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=1)

        matrix = builder.build_dataset(inst_id, bars, features, target)

        # Total bars = 10
        # Indices 0..3: SMA(5) returns None -> 4 dropped features
        # Index 9: Target has no future bar -> 1 dropped target
        # Indices 4..8: Complete rows -> 5 rows
        assert len(matrix) == 5
        assert matrix.dropped_incomplete_feature_count == 4
        assert matrix.dropped_incomplete_target_count == 1
        assert (
            len(matrix)
            + matrix.dropped_incomplete_feature_count
            + matrix.dropped_incomplete_target_count
            == 10
        )

    def test_insufficient_future_history_causes_row_drop_counted_once(
        self, builder: DatasetBuilder
    ) -> None:
        """9. Insufficient future history causes a row to be dropped and counted once."""
        inst_id = uuid4()
        # 10 bars
        closes = [str(100 + i) for i in range(10)]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)

        # 1d return requires 2 bars (indices 1..9 have valid feature)
        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d",
                parameters={"window": 1},
            ),
        ]
        # Target horizon 4 -> indices 6, 7, 8, 9 lack 4 future bars
        target = TargetDefinition(name="forward_return", horizon_bars=4)

        matrix = builder.build_dataset(inst_id, bars, features, target)

        # Index 0: missing feature -> dropped_incomplete_feature_count = 1
        # Indices 1..5: complete features and targets -> 5 rows
        # Indices 6..9: complete features, but missing target -> dropped_incomplete_target_count = 4
        assert len(matrix) == 5
        assert matrix.dropped_incomplete_feature_count == 1
        assert matrix.dropped_incomplete_target_count == 4

    def test_missing_requested_observation_dates_are_skipped(self, builder: DatasetBuilder) -> None:
        """10. Missing requested observation dates (e.g. weekends/holidays) are skipped."""
        inst_id = uuid4()
        # 5 bars on dates Jan 1, Jan 2, Jan 3, Jan 4, Jan 5
        closes = ["100", "102", "104", "106", "108"]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d",
                parameters={"window": 1},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=1)

        # Request dates: Jan 2 (present), Jan 10 (absent!), Jan 4 (present)
        req_dates = [date(2026, 1, 2), date(2026, 1, 10), date(2026, 1, 4)]
        matrix = builder.build_dataset(inst_id, bars, features, target, observation_dates=req_dates)

        # Only Jan 2 and Jan 4 are evaluated
        assert len(matrix) == 2
        assert matrix.observation_dates == (date(2026, 1, 2), date(2026, 1, 4))

    def test_unsorted_input_processed_chronologically_without_mutating_caller_sequence(
        self, builder: DatasetBuilder
    ) -> None:
        """11. Unsorted input is processed chronologically without mutating caller's sequence."""
        inst_id = uuid4()
        closes = ["100", "102", "104", "106", "108"]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)
        # Deliberately shuffle input bars
        shuffled_bars = [bars[3], bars[0], bars[4], bars[1], bars[2]]
        original_ids = [b.market_bar_id for b in shuffled_bars]

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d",
                parameters={"window": 1},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=1)

        matrix = builder.build_dataset(inst_id, shuffled_bars, features, target)

        # Caller's input sequence is not mutated
        assert [b.market_bar_id for b in shuffled_bars] == original_ids

        # Resulting matrix is strictly chronological
        assert matrix.observation_dates == (
            date(2026, 1, 2),
            date(2026, 1, 3),
            date(2026, 1, 4),
        )

    def test_interleaved_multi_instrument_input_isolation(self, builder: DatasetBuilder) -> None:
        """12. Interleaved bars from multiple instruments do not contaminate calculations."""
        inst_a = uuid4()
        inst_b = uuid4()

        bars_a = make_daily_series(date(2026, 1, 1), ["100", "102", "104", "106", "108"], inst_a)
        bars_b = make_daily_series(date(2026, 1, 1), ["500", "550", "600", "650", "700"], inst_b)

        combined = [
            bars_a[0],
            bars_b[0],
            bars_a[1],
            bars_b[1],
            bars_a[2],
            bars_b[2],
            bars_a[3],
            bars_b[3],
            bars_a[4],
            bars_b[4],
        ]

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d",
                parameters={"window": 1},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=1)

        matrix = builder.build_dataset(inst_a, combined, features, target)

        assert matrix.instrument_id == inst_a
        # Only bars of inst_a are used: return at Jan 2 is 102/100 - 1 = 0.02
        assert matrix.rows[0].feature_values[0] == Decimal("0.02")

    def test_duplicate_market_dates_for_requested_instrument_raises_value_error(
        self, builder: DatasetBuilder
    ) -> None:
        """13. Duplicate market dates for the requested instrument raise a clear ValueError."""
        inst_id = uuid4()
        ts = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
        bar1 = make_bar(ts, Decimal("100"), inst_id)
        bar2 = make_bar(ts + timedelta(hours=1), Decimal("102"), inst_id)  # Same date!

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d",
                parameters={"window": 1},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=1)

        with pytest.raises(ValueError, match="Duplicate market observation dates found"):
            builder.build_dataset(inst_id, [bar1, bar2], features, target)

    def test_insufficient_total_history_produces_valid_empty_matrix(
        self, builder: DatasetBuilder
    ) -> None:
        """14. Insufficient total history produces a valid empty matrix."""
        inst_id = uuid4()
        # Only 2 bars
        bars = make_daily_series(date(2026, 1, 1), ["100", "105"], inst_id)

        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="20d",
                parameters={"window": 20},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=5)

        matrix = builder.build_dataset(inst_id, bars, features, target)

        assert isinstance(matrix, FeatureMatrix)
        assert len(matrix) == 0
        assert matrix.rows == ()
        assert matrix.feature_matrix == ()
        assert matrix.target_vector == ()
        assert matrix.observation_dates == ()
        # Both bars lacked the 20d SMA -> 2 dropped features
        assert matrix.dropped_incomplete_feature_count == 2
        assert matrix.dropped_incomplete_target_count == 0

    def test_invalid_builder_inputs_rejected(self, builder: DatasetBuilder) -> None:
        """15. Invalid inputs to build_dataset are rejected."""
        inst_id = uuid4()
        bars = make_daily_series(date(2026, 1, 1), ["100"], inst_id)
        target = TargetDefinition(name="forward_return", horizon_bars=1)
        feat = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="1d",
            parameters={"window": 1},
        )

        # Invalid instrument_id
        with pytest.raises(TypeError, match="instrument_id"):
            builder.build_dataset("not-a-uuid", bars, [feat], target)  # type: ignore[arg-type]

        # Empty features
        with pytest.raises(ValueError, match="At least one FeatureDefinition"):
            builder.build_dataset(inst_id, bars, [], target)

        # Invalid features element
        with pytest.raises(TypeError, match="features\\[0\\]"):
            builder.build_dataset(inst_id, bars, ["not-a-feature"], target)  # type: ignore[list-item]

        # Invalid target
        with pytest.raises(TypeError, match="target"):
            builder.build_dataset(inst_id, bars, [feat], "not-a-target")  # type: ignore[arg-type]

        # Invalid bars element
        with pytest.raises(TypeError, match="bars\\[0\\]"):
            builder.build_dataset(inst_id, ["not-a-bar"], [feat], target)  # type: ignore[list-item]

        # Invalid observation_dates with datetime instead of date
        with pytest.raises(TypeError, match="must be a date, not a datetime"):
            builder.build_dataset(
                inst_id,
                bars,
                [feat],
                target,
                observation_dates=[datetime(2026, 1, 1)],
            )

        # Duplicate observation_dates requested
        with pytest.raises(ValueError, match="Duplicate observation dates"):
            builder.build_dataset(
                inst_id,
                bars,
                [feat],
                target,
                observation_dates=[date(2026, 1, 1), date(2026, 1, 1)],
            )

    def test_builder_drops_non_finite_features_and_preserves_finite_negative_returns(
        self,
    ) -> None:
        """Builder drops non-finite feature outputs (NaN/Infinity) into dropped_incomplete_feature_count

        while correctly preserving legitimate negative finite feature and target values.
        """
        from app.domain.features.feature_calculation_context import FeatureCalculationContext
        from app.domain.features.feature_calculator import FeatureCalculator
        from app.services.features.feature_calculator_registry import FeatureCalculatorRegistry

        inst_id = uuid4()
        # 5 daily bars with decreasing prices
        closes = ["100", "90", "80", "70", "60"]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)

        # 1. Test finite negative values preservation
        def_ret = FeatureDefinition(
            name="return",
            category=FeatureCategory.TECHNICAL,
            description="1d",
            parameters={"window": 1},
        )
        target = TargetDefinition(name="forward_return", horizon_bars=1)

        default_engine = FeatureEngine(create_default_feature_registry())
        normal_builder = DatasetBuilder(default_engine)
        matrix = normal_builder.build_dataset(
            inst_id, bars, [def_ret], target, observation_dates=[date(2026, 1, 2)]
        )

        assert len(matrix) == 1
        # Feature: (90 / 100) - 1 = -0.10 (finite negative)
        assert matrix.rows[0].feature_values[0] == Decimal("-0.10")
        # Target: (80 / 90) - 1 = -1/9 (finite negative)
        expected_target = (Decimal("80") / Decimal("90")) - Decimal("1")
        assert matrix.rows[0].target_value == expected_target

        # 2. Test non-finite feature output (e.g. Decimal('NaN')) is dropped
        class NanCalculator(FeatureCalculator):
            @property
            def feature_name(self) -> str:
                return "nan_feat"

            def calculate(
                self,
                definition: FeatureDefinition,
                context: FeatureCalculationContext,
            ) -> Decimal | None:
                return Decimal("NaN")

        reg = FeatureCalculatorRegistry()
        reg.register(NanCalculator())
        nan_engine = FeatureEngine(reg)
        nan_builder = DatasetBuilder(nan_engine)

        nan_feat = FeatureDefinition(
            name="nan_feat",
            category=FeatureCategory.TECHNICAL,
            description="nan",
        )
        nan_matrix = nan_builder.build_dataset(
            inst_id, bars, [nan_feat], target, observation_dates=[date(2026, 1, 2)]
        )
        assert len(nan_matrix) == 0
        assert nan_matrix.dropped_incomplete_feature_count == 1
        assert nan_matrix.dropped_incomplete_target_count == 0


class TestFeatureMatrixAndDatasetRowContract:
    def test_dataset_row_valid_creation_and_to_dict(self) -> None:
        inst_id = uuid4()
        row = DatasetRow(
            instrument_id=inst_id,
            observation_date=date(2026, 1, 5),
            realization_date=date(2026, 1, 12),
            feature_values=(Decimal("0.02"), Decimal("105.5")),
            target_value=Decimal("0.05"),
        )
        assert row.instrument_id == inst_id
        assert row.observation_date == date(2026, 1, 5)
        assert row.realization_date == date(2026, 1, 12)
        assert row.feature_values == (Decimal("0.02"), Decimal("105.5"))
        assert row.target_value == Decimal("0.05")

        as_dict = row.to_dict(["ret_1", "sma_5"])
        assert as_dict == {
            "instrument_id": inst_id,
            "observation_date": date(2026, 1, 5),
            "realization_date": date(2026, 1, 12),
            "target": Decimal("0.05"),
            "ret_1": Decimal("0.02"),
            "sma_5": Decimal("105.5"),
        }

    def test_dataset_row_validates_types_and_chronology(self) -> None:
        inst_id = uuid4()
        # Rejects datetime for observation_date
        with pytest.raises(TypeError, match="must be a date, not a datetime"):
            DatasetRow(
                instrument_id=inst_id,
                observation_date=datetime(2026, 1, 5),
                realization_date=date(2026, 1, 10),
                feature_values=(Decimal("1"),),
                target_value=Decimal("0.05"),
            )

        # Rejects realization_date <= observation_date
        with pytest.raises(ValueError, match="strictly after observation_date"):
            DatasetRow(
                instrument_id=inst_id,
                observation_date=date(2026, 1, 5),
                realization_date=date(2026, 1, 5),
                feature_values=(Decimal("1"),),
                target_value=Decimal("0.05"),
            )

        # Rejects non-Decimal in feature_values
        with pytest.raises(TypeError, match="Decimal instance"):
            DatasetRow(
                instrument_id=inst_id,
                observation_date=date(2026, 1, 5),
                realization_date=date(2026, 1, 6),
                feature_values=(1.23,),  # type: ignore[arg-type]
                target_value=Decimal("0.05"),
            )

    def test_dataset_row_rejects_non_finite_values_while_preserving_legitimate_negatives(
        self,
    ) -> None:
        """Reject Decimal('NaN'), Decimal('Infinity'), Decimal('-Infinity')

        while preserving finite negative values.
        """
        inst_id = uuid4()
        obs = date(2026, 1, 5)
        real = date(2026, 1, 6)

        # Legitimate finite negative values are valid
        valid_row = DatasetRow(
            instrument_id=inst_id,
            observation_date=obs,
            realization_date=real,
            feature_values=(Decimal("-0.05"), Decimal("-12.34")),
            target_value=Decimal("-0.10"),
        )
        assert valid_row.feature_values == (Decimal("-0.05"), Decimal("-12.34"))
        assert valid_row.target_value == Decimal("-0.10")

        # Reject non-finite feature values
        for non_finite in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with pytest.raises(ValueError, match="must be a finite Decimal"):
                DatasetRow(
                    instrument_id=inst_id,
                    observation_date=obs,
                    realization_date=real,
                    feature_values=(non_finite,),
                    target_value=Decimal("0.05"),
                )

        # Reject non-finite target value
        for non_finite in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with pytest.raises(ValueError, match="must be a finite Decimal"):
                DatasetRow(
                    instrument_id=inst_id,
                    observation_date=obs,
                    realization_date=real,
                    feature_values=(Decimal("0.05"),),
                    target_value=non_finite,
                )

    def test_feature_matrix_validations(self) -> None:
        """16. FeatureMatrix rejects inconsistent row widths, incorrect instrument IDs,

        duplicate observation dates, and non-chronological rows.
        """
        inst_id = uuid4()
        other_inst = uuid4()
        target = TargetDefinition(name="forward_return", horizon_bars=1)
        defn = FeatureDefinition(
            name="ret",
            category=FeatureCategory.TECHNICAL,
            description="1d",
            parameters={"window": 1},
        )

        row1 = DatasetRow(
            instrument_id=inst_id,
            observation_date=date(2026, 1, 5),
            realization_date=date(2026, 1, 6),
            feature_values=(Decimal("0.01"),),
            target_value=Decimal("0.02"),
        )
        row2 = DatasetRow(
            instrument_id=inst_id,
            observation_date=date(2026, 1, 6),
            realization_date=date(2026, 1, 7),
            feature_values=(Decimal("0.02"),),
            target_value=Decimal("0.03"),
        )

        # Valid matrix
        m = FeatureMatrix(
            instrument_id=inst_id,
            feature_names=("ret",),
            feature_definitions=(defn,),
            target_definition=target,
            rows=(row1, row2),
        )
        assert len(m) == 2

        # Inconsistent row width
        row_wrong_width = DatasetRow(
            instrument_id=inst_id,
            observation_date=date(2026, 1, 7),
            realization_date=date(2026, 1, 8),
            feature_values=(Decimal("0.01"), Decimal("0.02")),  # 2 values for 1 feature!
            target_value=Decimal("0.02"),
        )
        with pytest.raises(ValueError, match="feature_values width"):
            FeatureMatrix(
                instrument_id=inst_id,
                feature_names=("ret",),
                feature_definitions=(defn,),
                target_definition=target,
                rows=(row1, row_wrong_width),
            )

        # Incorrect instrument_id
        row_wrong_inst = DatasetRow(
            instrument_id=other_inst,
            observation_date=date(2026, 1, 7),
            realization_date=date(2026, 1, 8),
            feature_values=(Decimal("0.01"),),
            target_value=Decimal("0.02"),
        )
        with pytest.raises(ValueError, match="does not match matrix instrument_id"):
            FeatureMatrix(
                instrument_id=inst_id,
                feature_names=("ret",),
                feature_definitions=(defn,),
                target_definition=target,
                rows=(row1, row_wrong_inst),
            )

        # Duplicate observation_date
        row_duplicate_date = DatasetRow(
            instrument_id=inst_id,
            observation_date=date(2026, 1, 5),  # duplicate of row1!
            realization_date=date(2026, 1, 6),
            feature_values=(Decimal("0.01"),),
            target_value=Decimal("0.02"),
        )
        with pytest.raises(ValueError, match="duplicate observation_date"):
            FeatureMatrix(
                instrument_id=inst_id,
                feature_names=("ret",),
                feature_definitions=(defn,),
                target_definition=target,
                rows=(row1, row_duplicate_date),
            )

        # Non-chronological rows
        with pytest.raises(ValueError, match="must be chronologically ordered"):
            FeatureMatrix(
                instrument_id=inst_id,
                feature_names=("ret",),
                feature_definitions=(defn,),
                target_definition=target,
                rows=(row2, row1),  # Jan 6 before Jan 5!
            )

        # Reject booleans as drop counters
        with pytest.raises(TypeError, match="dropped_incomplete_feature_count"):
            FeatureMatrix(
                instrument_id=inst_id,
                feature_names=("ret",),
                feature_definitions=(defn,),
                target_definition=target,
                rows=(row1,),
                dropped_incomplete_feature_count=True,
            )

    def test_generate_feature_name_helper(self) -> None:
        defn_no_params = FeatureDefinition(
            name="return", category=FeatureCategory.TECHNICAL, description="ret"
        )
        assert generate_feature_name(defn_no_params) == "return"

        defn_with_params = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="bb",
            parameters={"window": 20, "band": "upper", "k": Decimal("2")},
        )
        # Keys sorted: band, k, window - ordinary scalar naming preserved
        assert generate_feature_name(defn_with_params) == "bollinger_bands_band_upper_k_2_window_20"

    def test_generate_feature_name_nested_structures_and_order_invariance(self) -> None:
        """Nested mappings produce identical column names regardless of dictionary insertion order,

        and distinct parameter values produce distinct column names.
        """
        # 1. Top-level insertion order invariance
        defn_top_1 = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="top1",
            parameters={"window": 10, "k": 2},
        )
        defn_top_2 = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="top2",
            parameters={"k": 2, "window": 10},
        )
        assert generate_feature_name(defn_top_1) == generate_feature_name(defn_top_2)
        assert generate_feature_name(defn_top_1) == "custom_k_2_window_10"

        # 2. Nested dictionary insertion order invariance
        defn_nested_1 = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="nested1",
            parameters={"config": {"beta": 2, "alpha": 1}},
        )
        defn_nested_2 = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="nested2",
            parameters={"config": {"alpha": 1, "beta": 2}},
        )
        assert generate_feature_name(defn_nested_1) == generate_feature_name(defn_nested_2)
        assert generate_feature_name(defn_nested_1) == "custom_config_{alpha:1, beta:2}"

        # 3. Complex nested structures (lists, tuples, nested dicts)
        defn_complex_1 = FeatureDefinition(
            name="complex",
            category=FeatureCategory.TECHNICAL,
            description="c1",
            parameters={
                "weights": [0.5, 0.5],
                "submap": {"y": {"b": 2, "a": 1}, "x": 10},
            },
        )
        defn_complex_2 = FeatureDefinition(
            name="complex",
            category=FeatureCategory.TECHNICAL,
            description="c2",
            parameters={
                "submap": {"x": 10, "y": {"a": 1, "b": 2}},
                "weights": [0.5, 0.5],
            },
        )
        assert generate_feature_name(defn_complex_1) == generate_feature_name(defn_complex_2)

        # 4. Distinct parameter structures produce distinct names
        defn_diff_val = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="diff_val",
            parameters={"config": {"alpha": 1, "beta": 999}},
        )
        assert generate_feature_name(defn_nested_1) != generate_feature_name(defn_diff_val)

        defn_diff_key = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="diff_key",
            parameters={"config": {"alpha": 1, "gamma": 2}},
        )
        assert generate_feature_name(defn_nested_1) != generate_feature_name(defn_diff_key)

    def test_generate_feature_name_collision_resistance(self) -> None:
        """Verify collision resistance across delimiter and type variations:

        - int vs str inside nested structures: {'x': 1} vs {'x': '1'}
        - underscore key vs value delimiter collisions: {'a_b': 'c'} vs {'a': 'b_c'}
        - list vs tuple container distinctions: [1, 2] vs (1, 2)
        - set vs frozenset container distinctions: {'a', 'b'} vs frozenset({'a', 'b'})
        - boolean vs integer vs string distinctions: True vs 1 vs 'True'
        - None vs string 'None'
        - Unsupported parameter types and keys raise TypeError / ValueError
        """
        # 1. int vs str in nested structures
        defn_int = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="int",
            parameters={"config": {"x": 1}},
        )
        defn_str = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="str",
            parameters={"config": {"x": "1"}},
        )
        name_int = generate_feature_name(defn_int)
        name_str = generate_feature_name(defn_str)
        assert name_int != name_str
        assert name_int == "custom_config_{x:1}"
        assert name_str == "custom_config_{x:'1'}"

        # 2. Top-level keys with underscores are rejected; values with underscores are safely escaped
        defn_key_under = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="key_under",
            parameters={"a_b": "c"},
        )
        with pytest.raises(
            ValueError,
            match="Top-level parameter key must be a single-word identifier without underscores",
        ):
            generate_feature_name(defn_key_under)

        defn_val_under = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="val_under",
            parameters={"a": "b_c"},
        )
        assert generate_feature_name(defn_val_under) == "custom_a_'b_c'"

        # Multi-parameter collision cannot occur because keys with underscores are rejected
        defn_multi_param = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="multi_param",
            parameters={"a": "b", "c": "d"},
        )
        assert generate_feature_name(defn_multi_param) == "custom_a_b_c_d"

        defn_colliding_key = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="colliding_key",
            parameters={"a_b_c": "d"},
        )
        with pytest.raises(
            ValueError,
            match="Top-level parameter key must be a single-word identifier without underscores",
        ):
            generate_feature_name(defn_colliding_key)

        # 3. List vs tuple distinction
        defn_list = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="list",
            parameters={"seq": [1, 2]},
        )
        defn_tuple = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="tuple",
            parameters={"seq": (1, 2)},
        )
        assert generate_feature_name(defn_list) != generate_feature_name(defn_tuple)
        assert generate_feature_name(defn_list) == "custom_seq_[1, 2]"
        assert generate_feature_name(defn_tuple) == "custom_seq_(1, 2)"

        # 4. Set vs frozenset distinction (with deterministic sorting)
        defn_set = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="set",
            parameters={"items": {"z", "a"}},
        )
        defn_frozenset = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="frozenset",
            parameters={"items": frozenset({"z", "a"})},
        )
        assert generate_feature_name(defn_set) != generate_feature_name(defn_frozenset)
        assert generate_feature_name(defn_set) == "custom_items_set{'a', 'z'}"
        assert generate_feature_name(defn_frozenset) == "custom_items_frozenset{'a', 'z'}"

        # 5. Boolean vs integer vs string distinction
        defn_bool = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="bool",
            parameters={"flag": True},
        )
        defn_int_1 = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="int_1",
            parameters={"flag": 1},
        )
        defn_str_true = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="str_true",
            parameters={"flag": "True"},
        )
        assert generate_feature_name(defn_bool) != generate_feature_name(defn_int_1)
        assert generate_feature_name(defn_bool) != generate_feature_name(defn_str_true)
        assert generate_feature_name(defn_int_1) != generate_feature_name(defn_str_true)
        assert generate_feature_name(defn_bool) == "custom_flag_True"
        assert generate_feature_name(defn_int_1) == "custom_flag_1"
        assert generate_feature_name(defn_str_true) == "custom_flag_'True'"

        # 6. None vs string 'None'
        defn_none = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="none",
            parameters={"val": None},
        )
        defn_str_none = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="str_none",
            parameters={"val": "None"},
        )
        assert generate_feature_name(defn_none) != generate_feature_name(defn_str_none)
        assert generate_feature_name(defn_none) == "custom_val_None"
        assert generate_feature_name(defn_str_none) == "custom_val_'None'"

        # 7. Date vs string date distinction
        defn_date = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="date_obj",
            parameters={"asof": date(2026, 1, 1)},
        )
        defn_str_date = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="date_str",
            parameters={"asof": "2026-01-01"},
        )
        assert generate_feature_name(defn_date) != generate_feature_name(defn_str_date)
        assert generate_feature_name(defn_date) == "custom_asof_2026-01-01"
        assert generate_feature_name(defn_str_date) == "custom_asof_'2026-01-01'"

        # 8. Non-finite float and Decimal parameter values are rejected
        defn_nan_float = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="nan_float",
            parameters={"k": float("nan")},
        )
        with pytest.raises(ValueError, match="Non-finite float"):
            generate_feature_name(defn_nan_float)

        defn_nan_decimal = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="nan_decimal",
            parameters={"k": Decimal("NaN")},
        )
        with pytest.raises(ValueError, match="Non-finite Decimal"):
            generate_feature_name(defn_nan_decimal)

        defn_inf_decimal = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="inf_decimal",
            parameters={"k": Decimal("Infinity")},
        )
        with pytest.raises(ValueError, match="Non-finite Decimal"):
            generate_feature_name(defn_inf_decimal)

        # 9. Unsupported parameter value types raise TypeError
        defn_unsupported_val = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="unsupported_val",
            parameters={"fn": lambda x: x},
        )
        with pytest.raises(TypeError, match="Unsupported parameter value type"):
            generate_feature_name(defn_unsupported_val)

        # 10. Unsupported top-level parameter key types, reserved words, or empty keys
        defn_empty_key = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="empty_key",
            parameters={"": "val"},
        )
        with pytest.raises(ValueError, match="Parameter key cannot be empty"):
            generate_feature_name(defn_empty_key)

        defn_int_key = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="int_key",
            parameters={123: "val"},  # type: ignore[dict-item]
        )
        with pytest.raises(TypeError, match="Parameter key must be a string"):
            generate_feature_name(defn_int_key)

        defn_hyphen_key = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="hyphen_key",
            parameters={"window-size": 10},
        )
        with pytest.raises(
            ValueError,
            match="Top-level parameter key must be a single-word identifier without underscores",
        ):
            generate_feature_name(defn_hyphen_key)

        defn_reserved_key = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="reserved_key",
            parameters={"True": 1},
        )
        with pytest.raises(
            ValueError,
            match="Top-level parameter key must be a single-word identifier without underscores",
        ):
            generate_feature_name(defn_reserved_key)

        # 11. Nested dictionary invalid keys
        defn_nested_bad_key = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="nested_bad_key",
            parameters={"config": {(1, 2): "val"}},
        )
        with pytest.raises(TypeError, match="Unsupported parameter dictionary key type"):
            generate_feature_name(defn_nested_bad_key)

        defn_nested_empty_key = FeatureDefinition(
            name="custom",
            category=FeatureCategory.TECHNICAL,
            description="nested_empty_key",
            parameters={"config": {"": "val"}},
        )
        with pytest.raises(ValueError, match="Nested parameter dictionary key cannot be empty"):
            generate_feature_name(defn_nested_empty_key)

        # 12. Integer and Decimal with equivalent textual representations intentionally share column name
        defn_k_int = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="k_int",
            parameters={"band": "upper", "k": 2, "window": 20},
        )
        defn_k_dec = FeatureDefinition(
            name="bollinger_bands",
            category=FeatureCategory.TECHNICAL,
            description="k_dec",
            parameters={"band": "upper", "k": Decimal("2"), "window": 20},
        )
        assert generate_feature_name(defn_k_int) == generate_feature_name(defn_k_dec)
        assert generate_feature_name(defn_k_int) == "bollinger_bands_band_upper_k_2_window_20"


class InMemoryCorporateActionRepository(CorporateActionRepository):
    """Simple in-memory repository for unit testing corporate action retrieval."""

    def __init__(self, actions: Sequence[CorporateAction] = ()) -> None:
        self.actions = list(actions)

    def get_by_id(self, corporate_action_id: UUID) -> CorporateAction | None:
        return next((a for a in self.actions if a.corporate_action_id == corporate_action_id), None)

    def get_by_instrument(
        self,
        instrument_id: UUID,
        start_date: date | None = None,
        end_date: date | None = None,
        action_type: CorporateActionType | None = None,
    ) -> list[CorporateAction]:
        res = [a for a in self.actions if a.instrument_id == instrument_id]
        if start_date is not None:
            res = [a for a in res if a.execution_date >= start_date]
        if end_date is not None:
            res = [a for a in res if a.execution_date <= end_date]
        if action_type is not None:
            res = [a for a in res if a.action_type == action_type]
        return res

    def get_by_instrument_and_date(
        self,
        instrument_id: UUID,
        execution_date: date,
        action_type: CorporateActionType | None = None,
    ) -> list[CorporateAction]:
        res = [
            a
            for a in self.actions
            if a.instrument_id == instrument_id and a.execution_date == execution_date
        ]
        if action_type is not None:
            res = [a for a in res if a.action_type == action_type]
        return res

    def save(self, action: CorporateAction) -> None:
        self.actions.append(action)


class TestDatasetBuilderTotalReturnIntegration:
    """Tests integration of TotalReturnCalculator and CorporateActionRepository into DatasetBuilder."""

    def test_forward_return_behavior_remains_unchanged(self, engine: FeatureEngine) -> None:
        """1. Existing forward_return targets retain raw price-return semantics."""
        inst_id = uuid4()
        closes = ["100", "200", "100"]  # t0=100, t1=200, t2=100
        bars = make_daily_series(date(2026, 1, 5), closes, inst_id)

        # Corporate actions in repo: a 2-for-1 split at t2
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 7),
            value=Decimal("2"),
        )
        repo = InMemoryCorporateActionRepository([split])

        builder_with_repo = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)
        builder_default = DatasetBuilder(feature_engine=engine)

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d return",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=1)

        matrix_repo = builder_with_repo.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target
        )
        matrix_default = builder_default.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target
        )

        # forward_return must completely ignore the corporate action and remain raw
        # Row 0: obs=t1(200), real=t2(100) -> raw return = 100/200 - 1 = -0.50
        assert len(matrix_repo) == 1
        assert len(matrix_default) == 1
        assert matrix_repo.target_vector[0] == Decimal("-0.5")
        assert matrix_default.target_vector[0] == Decimal("-0.5")
        assert matrix_repo.rows == matrix_default.rows

    def test_total_return_target_selection_invokes_total_return_calculator(
        self, engine: FeatureEngine
    ) -> None:
        """2. total_return target selection invokes TotalReturnCalculator."""
        inst_id = uuid4()
        bars = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("200"), inst_id),
            make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id),
        ]
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        repo = InMemoryCorporateActionRepository([split])
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)

        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)

        matrix = builder.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target
        )

        assert len(matrix) == 1
        # Economic value unchanged: 2-for-1 split produces 0.00 total return, not -0.50
        assert matrix.target_vector[0] == Decimal("0")

    def test_corporate_actions_filtered_to_target_horizon(self, engine: FeatureEngine) -> None:
        """3. Corporate actions are filtered strictly to (observation_date, realization_date]."""
        inst_id = uuid4()
        bars = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),  # t0
            make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("100"), inst_id),  # t1
            make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("95"), inst_id),  # t2
        ]
        # Action on t0 (excluded from (t0, t2])
        div_t0 = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 5),
            value=Decimal("10.00"),
        )
        # Action on t2 (inside (t0, t2])
        div_t2 = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 7),
            value=Decimal("5.00"),
        )
        repo = InMemoryCorporateActionRepository([div_t0, div_t2])
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)

        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=2)

        matrix = builder.build_dataset(
            instrument_id=inst_id,
            bars=bars,
            features=features,
            target=target,
            observation_dates=[date(2026, 1, 5)],
        )

        assert len(matrix) == 1
        # Div on t0 excluded. Div on t2 (5 on 95 close) offsets price drop from 100 to 95.
        # Total return = (95/100) * (1 + 5/95) - 1 = 0.00
        assert matrix.target_vector[0] == Decimal("0")

    def test_actions_after_realization_do_not_affect_label(self, engine: FeatureEngine) -> None:
        """4. Corporate actions executed after the realization date have zero effect on the label."""
        inst_id = uuid4()
        bars = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),  # t0
            make_bar(
                datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("105"), inst_id
            ),  # t1 (realization)
            make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("52.5"), inst_id),  # t2
        ]
        # Split on t2 (after t1 realization)
        future_split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 7),
            value=Decimal("2"),
        )
        repo = InMemoryCorporateActionRepository([future_split])
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)

        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)

        matrix = builder.build_dataset(
            instrument_id=inst_id,
            bars=bars,
            features=features,
            target=target,
            observation_dates=[date(2026, 1, 5)],
        )

        assert len(matrix) == 1
        # Target from t0 to t1 is 105/100 - 1 = 0.05. Future split at t2 has no effect.
        assert matrix.target_vector[0] == Decimal("0.05")

    def test_historical_feature_values_invariant_to_future_corporate_action_records(
        self, engine: FeatureEngine
    ) -> None:
        """5. Historical feature values are completely invariant to future corporate actions."""
        inst_id = uuid4()
        closes = ["100", "102", "104", "106", "108", "54", "55"]
        bars = make_daily_series(date(2026, 1, 5), closes, inst_id)

        # Split at index 5 (2026-01-10)
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 10),
            value=Decimal("2"),
        )
        repo_with_split = InMemoryCorporateActionRepository([split])
        repo_empty = InMemoryCorporateActionRepository([])

        builder_split = DatasetBuilder(
            feature_engine=engine, corporate_action_repository=repo_with_split
        )
        builder_clean = DatasetBuilder(
            feature_engine=engine, corporate_action_repository=repo_empty
        )

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d return",
                parameters={"window": 1},
            ),
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="3d SMA",
                parameters={"window": 3},
            ),
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)

        matrix_split = builder_split.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target
        )
        matrix_clean = builder_clean.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target
        )

        # Feature matrix values must be identical across both datasets
        assert len(matrix_split) == len(matrix_clean)
        for row_s, row_c in zip(matrix_split.rows, matrix_clean.rows, strict=True):
            assert row_s.feature_values == row_c.feature_values

    def test_actual_realization_dates_preserved(self, engine: FeatureEngine) -> None:
        """6. Actual discrete future trading-bar realization dates are preserved."""
        inst_id = uuid4()
        # Friday Jan 9, Monday Jan 12 (skipping weekend Jan 10-11)
        bars = [
            make_bar(datetime(2026, 1, 9, 10, tzinfo=UTC), Decimal("100"), inst_id),
            make_bar(datetime(2026, 1, 12, 10, tzinfo=UTC), Decimal("102"), inst_id),
        ]
        repo = InMemoryCorporateActionRepository([])
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)

        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)

        matrix = builder.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target
        )

        assert len(matrix) == 1
        assert matrix.rows[0].observation_date == date(2026, 1, 9)
        assert matrix.rows[0].realization_date == date(2026, 1, 12)

    def test_missing_corporate_action_data_follows_explicit_policy(
        self, engine: FeatureEngine
    ) -> None:
        """7. Missing corporate-action data follows an explicit policy (rejects unconfigured total_return; drops unrealizable rows)."""
        inst_id = uuid4()
        bars = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),
            make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("102"), inst_id),
        ]
        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)

        # Policy A: If no repository and no corporate_actions provided, raise ValueError
        builder_no_repo = DatasetBuilder(feature_engine=engine)
        with pytest.raises(
            ValueError,
            match="Target 'total_return' requires corporate actions",
        ):
            builder_no_repo.build_dataset(
                instrument_id=inst_id, bars=bars, features=features, target=target
            )

        # Policy B: If a dividend is recorded inside horizon but its ex-date bar is missing, drop row
        dividend = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.DIVIDEND,
            execution_date=date(2026, 1, 6),  # ex-date Jan 6
            value=Decimal("2.00"),
        )
        repo = InMemoryCorporateActionRepository([dividend])
        builder_with_repo = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)

        # Available bars: Jan 5 (t0) and Jan 7 (t1). Jan 6 bar is missing!
        bars_missing_ex = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),
            make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("102"), inst_id),
        ]
        matrix = builder_with_repo.build_dataset(
            instrument_id=inst_id,
            bars=bars_missing_ex,
            features=features,
            target=target,
            observation_dates=[date(2026, 1, 5)],
        )

        # The row is dropped because the target cannot be realized without fabricating ex-date prices
        assert len(matrix) == 0
        assert matrix.dropped_incomplete_target_count == 1

    def test_multiple_instruments_remain_isolated(self, engine: FeatureEngine) -> None:
        """8. Interleaved multi-instrument bars and actions remain strictly isolated."""
        inst_a = uuid4()
        inst_b = uuid4()

        bars = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_a),
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("500"), inst_b),
            make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("110"), inst_a),
            make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("250"), inst_b),
        ]

        # 2-for-1 split for instrument B only
        split_b = CorporateAction(
            instrument_id=inst_b,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 6),
            value=Decimal("2"),
        )
        repo = InMemoryCorporateActionRepository([split_b])
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)

        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)

        matrix_a = builder.build_dataset(
            instrument_id=inst_a, bars=bars, features=features, target=target
        )
        matrix_b = builder.build_dataset(
            instrument_id=inst_b, bars=bars, features=features, target=target
        )

        # Instrument A: (110 / 100) - 1 = 0.10 (not affected by split of B)
        assert len(matrix_a) == 1
        assert matrix_a.target_vector[0] == Decimal("0.1")

        # Instrument B: split applies -> 250 / (500 / 2) - 1 = 0.00
        assert len(matrix_b) == 1
        assert matrix_b.target_vector[0] == Decimal("0")

    def test_existing_chronological_split_purge_and_embargo_unchanged(
        self, engine: FeatureEngine
    ) -> None:
        """9. FeatureMatrix built with total_return integrates seamlessly with DatasetSplitter."""
        inst_id = uuid4()
        closes = [str(100 + i) for i in range(25)]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)

        repo = InMemoryCorporateActionRepository([])
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d return",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=3)

        matrix = builder.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target
        )

        splitter = DatasetSplitter()
        split_result = splitter.split(matrix, SplitConfig.default())

        assert len(split_result.train_rows) > 0
        assert len(split_result.val_rows) > 0
        assert len(split_result.test_rows) > 0
        assert split_result.retained_rows + split_result.total_removed_rows == len(matrix)

    def test_end_to_end_dataset_construction_both_target_types(self, engine: FeatureEngine) -> None:
        """10. End-to-end dataset construction comparing forward_return and total_return."""
        inst_id = uuid4()
        # 15 days of bars
        closes = [
            "100",
            "102",
            "104",
            "106",
            "108",
            "110",
            "112",
            "114",
            "57",
            "58",  # 2-for-1 split between bar 7 (114) and bar 8 (57)
            "59",
            "60",
            "61",
            "62",
            "63",
        ]
        bars = make_daily_series(date(2026, 1, 1), closes, inst_id)

        # Split on bar 8 date (2026-01-09)
        split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 9),
            value=Decimal("2"),
        )
        repo = InMemoryCorporateActionRepository([split])
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)

        features = [
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1d return",
                parameters={"window": 1},
            ),
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="3d SMA",
                parameters={"window": 3},
            ),
        ]

        target_fwd = TargetDefinition(name="forward_return", horizon_bars=1)
        target_tot = TargetDefinition(name="total_return", horizon_bars=1)

        matrix_fwd = builder.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target_fwd
        )
        matrix_tot = builder.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target_tot
        )

        assert len(matrix_fwd) == len(matrix_tot)
        assert matrix_fwd.feature_names == matrix_tot.feature_names

        # Compare row by row:
        for row_f, row_t in zip(matrix_fwd.rows, matrix_tot.rows, strict=True):
            assert row_f.observation_date == row_t.observation_date
            assert row_f.realization_date == row_t.realization_date
            assert row_f.feature_values == row_t.feature_values

            # On bar 7 (2026-01-08) pointing to bar 8 (2026-01-09):
            if row_f.observation_date == date(2026, 1, 8):
                # Raw forward return: 57 / 114 - 1 = -0.50 (looks like 50% crash)
                assert row_f.target_value == Decimal("-0.5")
                # Split-adjusted total return: 57 / (114 / 2) - 1 = 57 / 57 - 1 = 0.00
                assert row_t.target_value == Decimal("0")


class TestDatasetBuilderQueryBounding:
    """Tests verifying that the corporate-action repository receives the correct end_date bound."""

    def test_repository_receives_end_date_equal_to_last_bar_date(
        self, engine: FeatureEngine
    ) -> None:
        """Repository get_by_instrument is called with end_date=last_bar_date."""
        inst_id = uuid4()
        bars = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),
            make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("105"), inst_id),
            make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("110"), inst_id),
        ]

        # Track which kwargs get_by_instrument was called with
        captured_calls: list[dict] = []

        class TrackingRepository(CorporateActionRepository):
            def get_by_id(self, corporate_action_id: UUID) -> CorporateAction | None:
                return None

            def get_by_instrument(
                self,
                instrument_id: UUID,
                start_date: date | None = None,
                end_date: date | None = None,
                action_type=None,
            ) -> list[CorporateAction]:
                captured_calls.append(
                    {
                        "instrument_id": instrument_id,
                        "start_date": start_date,
                        "end_date": end_date,
                        "action_type": action_type,
                    }
                )
                return []

            def get_by_instrument_and_date(
                self,
                instrument_id: UUID,
                execution_date: date,
                action_type=None,
            ) -> list[CorporateAction]:
                return []

            def save(self, action: CorporateAction) -> None:
                pass

        repo = TrackingRepository()
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)
        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)
        builder.build_dataset(instrument_id=inst_id, bars=bars, features=features, target=target)

        assert len(captured_calls) == 1
        call = captured_calls[0]
        assert call["instrument_id"] == inst_id
        # end_date must equal the date of the last available bar
        assert call["end_date"] == date(2026, 1, 7)
        # start_date is not bounded (remains None; the whole history is valid for target windows)
        assert call["start_date"] is None

    def test_actions_after_last_bar_are_not_required_for_dataset(
        self, engine: FeatureEngine
    ) -> None:
        """Corporate actions whose execution_date is after the last bar have no effect on labels."""
        inst_id = uuid4()
        bars = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),
            make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("105"), inst_id),
        ]

        # Split scheduled well after the last available bar — must be excluded by end_date bounding
        future_split = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 3, 1),  # far after last bar (2026-01-06)
            value=Decimal("2"),
        )
        # InMemoryCorporateActionRepository honours end_date filtering
        repo = InMemoryCorporateActionRepository([future_split])
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)

        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)
        matrix = builder.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target
        )

        assert len(matrix) == 1
        # No split inside the horizon: total return = (105/100) - 1 = 0.05
        assert matrix.target_vector[0] == Decimal("0.05")

    def test_forward_return_target_does_not_query_repository(self, engine: FeatureEngine) -> None:
        """forward_return callers with no repository remain fully functional (no repo query)."""
        inst_id = uuid4()
        bars = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),
            make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("110"), inst_id),
        ]
        builder = DatasetBuilder(feature_engine=engine)  # no repository
        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=1)
        matrix = builder.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target
        )
        assert len(matrix) == 1
        assert matrix.target_vector[0] == Decimal("0.1")

    def test_empty_bars_does_not_query_repository(self, engine: FeatureEngine) -> None:
        """Empty bar input avoids querying the repository entirely (no unbounded end_date=None lookup)."""
        inst_id = uuid4()
        other_id = uuid4()
        captured_calls: list[dict] = []

        class TrackingRepository(CorporateActionRepository):
            def get_by_id(self, corporate_action_id: UUID) -> CorporateAction | None:
                return None

            def get_by_instrument(
                self,
                instrument_id: UUID,
                start_date: date | None = None,
                end_date: date | None = None,
                action_type=None,
            ) -> list[CorporateAction]:
                captured_calls.append({"instrument_id": instrument_id, "end_date": end_date})
                return []

            def get_by_instrument_and_date(
                self,
                instrument_id: UUID,
                execution_date: date,
                action_type=None,
            ) -> list[CorporateAction]:
                return []

            def save(self, action: CorporateAction) -> None:
                pass

        repo = TrackingRepository()
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)
        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)

        # 1. Completely empty bars sequence
        matrix_empty = builder.build_dataset(
            instrument_id=inst_id, bars=[], features=features, target=target
        )
        assert len(matrix_empty) == 0
        assert len(captured_calls) == 0, "Repository must NOT be queried when bars is empty"

        # 2. Bars present only for a different instrument
        bars_other = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), other_id),
        ]
        matrix_other = builder.build_dataset(
            instrument_id=inst_id, bars=bars_other, features=features, target=target
        )
        assert len(matrix_other) == 0
        assert len(captured_calls) == 0, (
            "Repository must NOT be queried when no bars exist for instrument"
        )

    def test_empty_bars_with_unconfigured_source_raises_error(self, engine: FeatureEngine) -> None:
        """Unconfigured corporate action source raises ValueError even when bars is empty."""
        inst_id = uuid4()
        builder = DatasetBuilder(feature_engine=engine)  # no repository
        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)

        with pytest.raises(
            ValueError,
            match="Target 'total_return' requires corporate actions",
        ):
            builder.build_dataset(instrument_id=inst_id, bars=[], features=features, target=target)

    def test_action_after_row_realization_cannot_affect_label_even_if_fetched_for_dataset(
        self, engine: FeatureEngine
    ) -> None:
        """An action within the dataset range (fetched) cannot affect an earlier row whose realization preceded it."""
        inst_id = uuid4()
        # 3 daily bars: t0=Jan 5 (100), t1=Jan 6 (105), t2=Jan 7 (52.5)
        # Horizon = 1 bar:
        #   Row 0: obs=Jan 5, real=Jan 6
        #   Row 1: obs=Jan 6, real=Jan 7
        bars = [
            make_bar(datetime(2026, 1, 5, 10, tzinfo=UTC), Decimal("100"), inst_id),
            make_bar(datetime(2026, 1, 6, 10, tzinfo=UTC), Decimal("105"), inst_id),
            make_bar(datetime(2026, 1, 7, 10, tzinfo=UTC), Decimal("52.5"), inst_id),
        ]

        # 2:1 split on Jan 7 (inside the overall dataset bounds, so fetched with end_date=2026-01-07)
        split_jan7 = CorporateAction(
            instrument_id=inst_id,
            action_type=CorporateActionType.SPLIT,
            execution_date=date(2026, 1, 7),
            value=Decimal("2"),
        )
        repo = InMemoryCorporateActionRepository([split_jan7])
        builder = DatasetBuilder(feature_engine=engine, corporate_action_repository=repo)

        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="1d SMA",
                parameters={"window": 1},
            )
        ]
        target = TargetDefinition(name="total_return", horizon_bars=1)

        matrix = builder.build_dataset(
            instrument_id=inst_id, bars=bars, features=features, target=target
        )

        assert len(matrix) == 2

        # Row 0 (Jan 5 -> Jan 6): Realization is Jan 6. Split is Jan 7.
        # Split is AFTER realization date -> must NOT affect Row 0.
        # Raw return: 105 / 100 - 1 = 0.05
        assert matrix.rows[0].observation_date == date(2026, 1, 5)
        assert matrix.rows[0].realization_date == date(2026, 1, 6)
        assert matrix.rows[0].target_value == Decimal("0.05")

        # Row 1 (Jan 6 -> Jan 7): Realization is Jan 7. Split is Jan 7.
        # Split is ON realization date -> applies to Row 1.
        # Adjusted return: 52.5 / (105 / 2) - 1 = 52.5 / 52.5 - 1 = 0.00
        assert matrix.rows[1].observation_date == date(2026, 1, 6)
        assert matrix.rows[1].realization_date == date(2026, 1, 7)
        assert matrix.rows[1].target_value == Decimal("0")


class TestDatasetBuilderConstructor:
    """Tests verifying the constructor parameter contract and collaborator isolation."""

    def test_default_construction(self, engine: FeatureEngine) -> None:
        """Default constructor instantiates standard ForwardReturnCalculator and TotalReturnCalculator."""
        from app.domain.prediction.calculators.forward_return_calculator import (
            ForwardReturnCalculator,
        )
        from app.domain.prediction.calculators.total_return_calculator import (
            TotalReturnCalculator,
        )

        builder = DatasetBuilder(feature_engine=engine)
        assert builder.feature_engine is engine
        assert isinstance(builder.target_calculator, ForwardReturnCalculator)
        assert isinstance(builder.total_return_calculator, TotalReturnCalculator)
        assert builder.corporate_action_repository is None

    def test_explicit_collaborators_injection(self, engine: FeatureEngine) -> None:
        """Explicit collaborators are bound directly to their corresponding properties."""
        from app.domain.prediction.calculators.forward_return_calculator import (
            ForwardReturnCalculator,
        )
        from app.domain.prediction.calculators.total_return_calculator import (
            TotalReturnCalculator,
        )

        fwd_calc = ForwardReturnCalculator()
        tot_calc = TotalReturnCalculator()
        repo = InMemoryCorporateActionRepository([])

        builder = DatasetBuilder(
            feature_engine=engine,
            target_calculator=fwd_calc,
            corporate_action_repository=repo,
            total_return_calculator=tot_calc,
        )
        assert builder.target_calculator is fwd_calc
        assert builder.total_return_calculator is tot_calc
        assert builder.corporate_action_repository is repo

    def test_parameter_isolation(self, engine: FeatureEngine) -> None:
        """Injecting target_calculator or total_return_calculator isolates each to its own property."""
        from app.domain.prediction.calculators.forward_return_calculator import (
            ForwardReturnCalculator,
        )
        from app.domain.prediction.calculators.total_return_calculator import (
            TotalReturnCalculator,
        )

        fwd_calc = ForwardReturnCalculator()
        builder_fwd = DatasetBuilder(feature_engine=engine, target_calculator=fwd_calc)
        assert builder_fwd.target_calculator is fwd_calc
        assert isinstance(builder_fwd.total_return_calculator, TotalReturnCalculator)
        assert builder_fwd.total_return_calculator is not fwd_calc

        tot_calc = TotalReturnCalculator()
        builder_tot = DatasetBuilder(feature_engine=engine, total_return_calculator=tot_calc)
        assert builder_tot.total_return_calculator is tot_calc
        assert isinstance(builder_tot.target_calculator, ForwardReturnCalculator)
        assert builder_tot.target_calculator is not tot_calc

    def test_type_validation_rejects_invalid_collaborators(self, engine: FeatureEngine) -> None:
        """Constructor validates collaborator types and rejects invalid inputs with TypeError."""
        with pytest.raises(TypeError, match="feature_engine must be a FeatureEngine instance"):
            DatasetBuilder(feature_engine="not_an_engine")  # type: ignore[arg-type]

        with pytest.raises(
            TypeError, match="target_calculator must be a ForwardReturnCalculator instance"
        ):
            DatasetBuilder(feature_engine=engine, target_calculator="not_a_calc")  # type: ignore[arg-type]

        with pytest.raises(
            TypeError,
            match="corporate_action_repository must be a CorporateActionRepository instance",
        ):
            DatasetBuilder(feature_engine=engine, corporate_action_repository="not_a_repo")  # type: ignore[arg-type]

        with pytest.raises(
            TypeError, match="total_return_calculator must be a TotalReturnCalculator instance"
        ):
            DatasetBuilder(feature_engine=engine, total_return_calculator="not_a_calc")  # type: ignore[arg-type]
