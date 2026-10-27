"""Unit tests for Phase 5.3 - DatasetValidator, DatasetSplitter, SplitConfig, SplitResult."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, timedelta
from decimal import Decimal
from uuid import UUID

import pytest

from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.prediction.dataset import DatasetRow, FeatureMatrix
from app.domain.prediction.split_config import SplitConfig
from app.domain.prediction.target_definition import TargetDefinition
from app.services.prediction.dataset_splitter import DatasetSplitter, DatasetValidator

# ===========================================================================
# Helpers & Fixtures
# ===========================================================================

_INST_ID: UUID = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
_HORIZON = 3  # 3-bar forward return horizon for all tests


def _make_feature_def(name: str = "sma") -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        parameters={"window": 5},
        category=FeatureCategory.TECHNICAL,
        description="Simple moving average",
    )


def _make_target_def(horizon: int = _HORIZON) -> TargetDefinition:
    return TargetDefinition(name="forward_return", horizon_bars=horizon)


def _make_row(
    obs_date: date,
    realization_date: date | None = None,
    instrument_id: UUID | None = None,
    fv: Decimal = Decimal("1.5"),
    tv: Decimal = Decimal("0.02"),
) -> DatasetRow:
    inst = instrument_id or _INST_ID
    real = realization_date or (obs_date + timedelta(days=_HORIZON))
    return DatasetRow(
        instrument_id=inst,
        observation_date=obs_date,
        realization_date=real,
        feature_values=(fv,),
        target_value=tv,
    )


def _make_matrix(
    n: int,
    start: date | None = None,
    horizon: int = _HORIZON,
    instrument_id: UUID | None = None,
) -> FeatureMatrix:
    """Build a FeatureMatrix with *n* daily rows starting from *start*."""
    d0 = start or date(2024, 1, 1)
    inst = instrument_id or _INST_ID
    rows = []
    for i in range(n):
        obs = d0 + timedelta(days=i)
        real = obs + timedelta(days=horizon)
        rows.append(
            DatasetRow(
                instrument_id=inst,
                observation_date=obs,
                realization_date=real,
                feature_values=(Decimal("1") + Decimal(str(i)) * Decimal("0.01"),),
                target_value=Decimal("0.01"),
            )
        )
    return FeatureMatrix(
        instrument_id=inst,
        feature_names=("sma_window_5",),
        feature_definitions=(_make_feature_def(),),
        target_definition=_make_target_def(horizon),
        rows=tuple(rows),
    )


@pytest.fixture
def validator() -> DatasetValidator:
    return DatasetValidator()


@pytest.fixture
def splitter() -> DatasetSplitter:
    return DatasetSplitter()


@pytest.fixture
def config_70_15_15() -> SplitConfig:
    return SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)


# ===========================================================================
# SplitConfig tests
# ===========================================================================


class TestSplitConfig:
    def test_valid_fractions(self) -> None:
        cfg = SplitConfig(train_fraction=0.7, val_fraction=0.15, test_fraction=0.15)
        assert cfg.train_fraction == 0.7
        assert cfg.val_fraction == 0.15
        assert cfg.test_fraction == 0.15
        assert cfg.purge_overlap is True
        assert cfg.embargo_bars == 0

    def test_default_classmethod(self) -> None:
        cfg = SplitConfig.default()
        assert abs(cfg.train_fraction - 0.70) < 1e-12
        assert abs(cfg.val_fraction - 0.15) < 1e-12
        assert abs(cfg.test_fraction - 0.15) < 1e-12

    def test_fractions_must_sum_to_one(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            SplitConfig(train_fraction=0.7, val_fraction=0.2, test_fraction=0.2)

    def test_fraction_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
            SplitConfig(train_fraction=0.0, val_fraction=0.5, test_fraction=0.5)

    def test_fraction_one_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
            SplitConfig(train_fraction=1.0, val_fraction=0.0, test_fraction=0.0)

    def test_fraction_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
            SplitConfig(train_fraction=-0.1, val_fraction=0.6, test_fraction=0.5)

    def test_fraction_must_be_float_not_int(self) -> None:
        with pytest.raises(TypeError, match="must be a float"):
            SplitConfig(train_fraction=1, val_fraction=0.15, test_fraction=0.15)

    def test_fraction_must_be_float_not_bool(self) -> None:
        with pytest.raises(TypeError, match="must be a float"):
            SplitConfig(train_fraction=True, val_fraction=0.15, test_fraction=0.15)

    def test_embargo_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SplitConfig(train_fraction=0.7, val_fraction=0.15, test_fraction=0.15, embargo_bars=-1)

    def test_embargo_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be an int"):
            SplitConfig(
                train_fraction=0.7,
                val_fraction=0.15,
                test_fraction=0.15,
                embargo_bars=True,
            )

    def test_purge_overlap_non_bool_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a bool"):
            SplitConfig(
                train_fraction=0.7,
                val_fraction=0.15,
                test_fraction=0.15,
                purge_overlap=1,  # type: ignore[arg-type]
            )

    def test_custom_embargo_and_no_purge(self) -> None:
        cfg = SplitConfig(
            train_fraction=0.6,
            val_fraction=0.2,
            test_fraction=0.2,
            purge_overlap=False,
            embargo_bars=5,
        )
        assert cfg.embargo_bars == 5
        assert cfg.purge_overlap is False

    def test_frozen(self) -> None:
        cfg = SplitConfig.default()
        with pytest.raises(FrozenInstanceError):
            cfg.train_fraction = 0.5  # type: ignore[misc]

    def test_tolerance_boundary_accepted(self) -> None:
        # Floating-point arithmetic means 0.1 + 0.2 + 0.7 is not exactly 1.0
        # but must be within tolerance.
        cfg = SplitConfig(train_fraction=0.1, val_fraction=0.2, test_fraction=0.7)
        assert cfg.train_fraction == 0.1


# ===========================================================================
# DatasetValidator tests
# ===========================================================================


class TestDatasetValidator:
    def test_valid_matrix_passes(self, validator: DatasetValidator) -> None:
        m = _make_matrix(10)
        validator.validate(m)  # must not raise

    def test_non_matrix_type_raises_type_error(self, validator: DatasetValidator) -> None:
        with pytest.raises(TypeError, match="FeatureMatrix"):
            validator.validate("not a matrix")  # type: ignore[arg-type]

    def test_empty_matrix_rejected(self, validator: DatasetValidator) -> None:
        m = FeatureMatrix(
            instrument_id=_INST_ID,
            feature_names=("sma_window_5",),
            feature_definitions=(_make_feature_def(),),
            target_definition=_make_target_def(),
            rows=(),
        )
        with pytest.raises(ValueError, match="at least one row"):
            validator.validate(m)

    def test_single_row_is_valid(self, validator: DatasetValidator) -> None:
        m = _make_matrix(1)
        validator.validate(m)  # must not raise

    def test_non_chronological_rows_rejected(self, validator: DatasetValidator) -> None:
        # FeatureMatrix itself rejects non-chronological rows on construction.
        obs1 = date(2024, 1, 5)
        obs2 = date(2024, 1, 3)  # earlier than obs1 — wrong order
        with pytest.raises(ValueError):
            FeatureMatrix(
                instrument_id=_INST_ID,
                feature_names=("sma_window_5",),
                feature_definitions=(_make_feature_def(),),
                target_definition=_make_target_def(),
                rows=(
                    _make_row(obs1),
                    _make_row(obs2),
                ),
            )

    def test_duplicate_observation_dates_rejected(self, validator: DatasetValidator) -> None:
        obs = date(2024, 1, 1)
        # FeatureMatrix itself rejects duplicates on construction.
        with pytest.raises(ValueError):
            FeatureMatrix(
                instrument_id=_INST_ID,
                feature_names=("sma_window_5",),
                feature_definitions=(_make_feature_def(),),
                target_definition=_make_target_def(),
                rows=(
                    _make_row(obs),
                    _make_row(obs),
                ),
            )


# ===========================================================================
# DatasetSplitter - basic allocation tests
# ===========================================================================


class TestDatasetSplitterAllocation:
    def test_basic_70_15_15_split(self, splitter: DatasetSplitter) -> None:
        # 20 rows: train=14, val=3, test=3 with default 70/15/15 and no purge needed.
        # horizon=0 would be invalid, use horizon=1 so realization = obs + 1 day.
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.total_rows == 20
        # train=floor(0.70*20)=14, val=floor(0.15*20)=3, test=20-14-3=3
        # With horizon=1: last train row's realization_date = obs_date + 1 day.
        # First val row obs_date = day 14. Last train realization = day 13 + 1 = day 14.
        # Purge condition: obs_date < realization_date -> day14 < day14 -> False -> not purged.
        assert result.train_size == 14
        assert result.val_rows_purged == 0
        assert result.val_rows_embargoed == 0
        assert result.test_rows_purged == 0
        assert result.test_rows_embargoed == 0

    def test_rounding_remainder_goes_to_test(self, splitter: DatasetSplitter) -> None:
        # 10 rows with 60/20/20:
        # train=floor(0.6*10)=6, val=floor(0.2*10)=2, test=10-6-2=2 (exact)
        m = _make_matrix(10, horizon=1)
        cfg = SplitConfig(train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)
        result = splitter.split(m, cfg)
        assert result.total_rows == 10
        assert result.train_size + result.val_size + result.test_size <= 10

    def test_rounding_remainder_example(self, splitter: DatasetSplitter) -> None:
        # 7 rows with 70/15/15:
        # train=floor(0.70*7)=4, val=floor(0.15*7)=1, test=7-4-1=2
        m = _make_matrix(7, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.total_rows == 7
        train_alloc = int(0.70 * 7)
        val_alloc = int(0.15 * 7)
        test_alloc = 7 - train_alloc - val_alloc
        assert (
            result.train_size + result.val_size + result.test_size
            == (train_alloc + val_alloc + test_alloc)
            - result.val_rows_purged
            - result.val_rows_embargoed
            - result.test_rows_purged
            - result.test_rows_embargoed
        )

    def test_config_type_error(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(10)
        with pytest.raises(TypeError, match="SplitConfig"):
            splitter.split(m, "bad")  # type: ignore[arg-type]

    def test_matrix_type_error(self, splitter: DatasetSplitter) -> None:
        cfg = SplitConfig.default()
        with pytest.raises(TypeError, match="FeatureMatrix"):
            splitter.split("bad", cfg)  # type: ignore[arg-type]

    def test_too_few_rows_for_val_raises(self, splitter: DatasetSplitter) -> None:
        # 2 rows: val = floor(0.15*2) = 0
        m = _make_matrix(2, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        with pytest.raises(ValueError, match="allocates 0 rows"):
            splitter.split(m, cfg)

    def test_chronological_order_preserved_in_result(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)

        def is_sorted(rows: tuple[DatasetRow, ...]) -> bool:
            return all(
                rows[i].observation_date < rows[i + 1].observation_date
                for i in range(len(rows) - 1)
            )

        assert is_sorted(result.train_rows)
        if len(result.val_rows) > 1:
            assert is_sorted(result.val_rows)
        if len(result.test_rows) > 1:
            assert is_sorted(result.test_rows)

    def test_no_row_duplication_across_splits(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        all_obs = (
            [r.observation_date for r in result.train_rows]
            + [r.observation_date for r in result.val_rows]
            + [r.observation_date for r in result.test_rows]
        )
        assert len(all_obs) == len(set(all_obs))

    def test_train_rows_always_precede_val(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        if result.train_rows and result.val_rows:
            assert result.train_rows[-1].observation_date < result.val_rows[0].observation_date

    def test_val_rows_precede_test_rows(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        if result.val_rows and result.test_rows:
            assert result.val_rows[-1].observation_date < result.test_rows[0].observation_date


# ===========================================================================
# DatasetSplitter - purge overlap tests
# ===========================================================================


class TestDatasetSplitterPurge:
    """Tests for target-interval overlap purging at split boundaries."""

    def _make_matrix_with_horizon(self, n: int, horizon: int) -> FeatureMatrix:
        return _make_matrix(n, horizon=horizon)

    def test_purge_removes_overlapping_val_rows(self, splitter: DatasetSplitter) -> None:
        """With horizon=5, the last train row's realization reaches 5 days ahead.
        Val rows whose observation_date < that realization_date must be purged."""
        # 30 rows, horizon=5
        # train=floor(0.7*30)=21, val=floor(0.15*30)=4, test=30-21-4=5
        # Last train row: obs=day20 (0-indexed), real=day20+5=day25.
        # Val starts at day21. Purge condition: obs_date < day25.
        # day21 < day25 -> purge. day22 < day25 -> purge. day23 < day25 -> purge.
        # day24 < day25 -> purge. So all 4 val rows are purged.
        m = self._make_matrix_with_horizon(30, horizon=5)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.val_rows_purged == 4  # all 4 raw val rows purged
        assert result.val_size == 0

    def test_purge_disabled_keeps_all_val_rows(self, splitter: DatasetSplitter) -> None:
        m = self._make_matrix_with_horizon(30, horizon=5)
        cfg = SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            purge_overlap=False,
        )
        result = splitter.split(m, cfg)
        assert result.val_rows_purged == 0
        assert result.val_size == 4  # raw val_size = floor(0.15*30) = 4

    def test_no_purge_when_horizon_equals_one(self, splitter: DatasetSplitter) -> None:
        """Horizon=1: last train real = obs+1. First val obs = obs+1. obs+1 < obs+1 is False."""
        # No row should be purged.
        m = self._make_matrix_with_horizon(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.val_rows_purged == 0
        assert result.test_rows_purged == 0

    def test_partial_purge(self, splitter: DatasetSplitter) -> None:
        """Test that only the overlapping head rows are purged, not all val rows.

        Horizon=3, daily observations:
        train=floor(0.7*30)=21 rows (obs day0..day20), last train real = day20+3 = day23.
        val_raw starts at day21. Rows day21, day22 are < day23 -> purged (2 rows).
        Row day23: day23 < day23 is False -> kept. val_raw has 4 rows: 2 purged, 2 kept.
        """
        m = self._make_matrix_with_horizon(30, horizon=3)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        # val_raw = rows[21:25], last_train_realization = day20 + 3 = day23
        # Purge: day21 < day23 yes, day22 < day23 yes, day23 < day23 no -> 2 purged
        assert result.val_rows_purged == 2
        assert result.val_size == 2

    def test_purge_count_is_zero_when_no_overlap(self, splitter: DatasetSplitter) -> None:
        m = self._make_matrix_with_horizon(30, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.val_rows_purged == 0

    def test_test_purge_uses_last_val_boundary(self, splitter: DatasetSplitter) -> None:
        """After purging, test purge boundary is last retained val row's realization_date."""
        # horizon=3, 30 rows:
        # train=21, val_raw=[21..24], test_raw=[25..29]
        # After partial purge: val_rows = [23, 24] (obs_dates).
        # Last val realization = day24+3 = day27.
        # test_raw starts at day25. Purge: day25 < day27 yes, day26 < day27 yes, day27 < day27 no.
        # 2 test rows purged.
        m = self._make_matrix_with_horizon(30, horizon=3)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.test_rows_purged == 2

    def test_test_boundary_falls_back_to_train_when_val_empty(
        self, splitter: DatasetSplitter
    ) -> None:
        """When all val rows are purged, test boundary uses last train row."""
        # horizon=5, 30 rows: all 4 val rows purged.
        # Last train real = day20+5 = day25.
        # test_raw starts at day25. Purge: day25 < day25 is False -> no test rows purged.
        m = self._make_matrix_with_horizon(30, horizon=5)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.val_size == 0
        assert result.val_rows_purged == 4
        # test fallback: boundary = last train row, real=day25, test starts at day25
        # day25 < day25 is False -> 0 purged
        assert result.test_rows_purged == 0

    def test_purge_with_calendar_weekend_and_holiday_gaps(self, splitter: DatasetSplitter) -> None:
        """Purging across non-contiguous observation dates (weekend and holiday gaps).

        Realistic schedule with fixed horizon_bars=2 trading bars:
          0: 2024-01-04 (Thu) -> real: 2024-01-08 (Mon)
          1: 2024-01-05 (Fri) -> real: 2024-01-09 (Tue) [spans weekend Jan 6-7]
          2: 2024-01-08 (Mon) -> real: 2024-01-10 (Wed)
          3: 2024-01-09 (Tue) -> real: 2024-01-11 (Thu)
          4: 2024-01-10 (Wed) -> real: 2024-01-12 (Fri)
          5: 2024-01-11 (Thu) -> real: 2024-01-16 (Tue) [spans weekend Jan 13-14 + MLK Day Jan 15]
          6: 2024-01-12 (Fri) -> real: 2024-01-17 (Wed) [spans weekend Jan 13-14 + MLK Day Jan 15]
          7: 2024-01-16 (Tue) -> real: 2024-01-18 (Thu)
          8: 2024-01-17 (Wed) -> real: 2024-01-19 (Fri)
          9: 2024-01-18 (Thu) -> real: 2024-01-22 (Mon) [spans weekend Jan 20-21]

        Split: 20% train (2 rows), 50% val (5 rows), 30% test (3 rows).
        - train_raw: [Thu Jan 4, Fri Jan 5]. Last train real = Tue Jan 9.
        - val_raw: [Mon Jan 8, Tue Jan 9, Wed Jan 10, Thu Jan 11, Fri Jan 12].
          * Mon Jan 8 < Tue Jan 9 -> PURGED (overlaps target interval of Fri Jan 5).
          * Tue Jan 9 < Tue Jan 9 is False -> RETAINED (boundary equality under (obs, real]).
          * Wed Jan 10, Thu Jan 11, Fri Jan 12 -> RETAINED.
          * val_rows = [Tue Jan 9, Wed Jan 10, Thu Jan 11, Fri Jan 12].
          * val_rows_purged = 1.
          * Last retained val row: Fri Jan 12. Realization = Wed Jan 17.
        - test_raw: [Tue Jan 16, Wed Jan 17, Thu Jan 18].
          * Tue Jan 16 < Wed Jan 17 -> PURGED (overlaps target interval of Fri Jan 12).
          * Wed Jan 17 < Wed Jan 17 is False -> RETAINED (boundary equality).
          * Thu Jan 18 -> RETAINED.
          * test_rows = [Wed Jan 17, Thu Jan 18].
          * test_rows_purged = 1.
        """
        trading_days = [
            date(2024, 1, 4),  # 0: Thu
            date(2024, 1, 5),  # 1: Fri
            date(2024, 1, 8),  # 2: Mon
            date(2024, 1, 9),  # 3: Tue
            date(2024, 1, 10),  # 4: Wed
            date(2024, 1, 11),  # 5: Thu
            date(2024, 1, 12),  # 6: Fri
            date(2024, 1, 16),  # 7: Tue (MLK Day holiday on Jan 15)
            date(2024, 1, 17),  # 8: Wed
            date(2024, 1, 18),  # 9: Thu
        ]
        realization_days = [
            date(2024, 1, 8),  # 0 -> 2
            date(2024, 1, 9),  # 1 -> 3
            date(2024, 1, 10),  # 2 -> 4
            date(2024, 1, 11),  # 3 -> 5
            date(2024, 1, 12),  # 4 -> 6
            date(2024, 1, 16),  # 5 -> 7
            date(2024, 1, 17),  # 6 -> 8
            date(2024, 1, 18),  # 7 -> 9
            date(2024, 1, 19),  # 8 -> Fri Jan 19
            date(2024, 1, 22),  # 9 -> Mon Jan 22
        ]

        rows = tuple(
            DatasetRow(
                instrument_id=_INST_ID,
                observation_date=obs,
                realization_date=real,
                feature_values=(Decimal("100.00"),),
                target_value=Decimal("0.0150"),
            )
            for obs, real in zip(trading_days, realization_days, strict=True)
        )
        matrix = FeatureMatrix(
            instrument_id=_INST_ID,
            feature_names=("sma_window_5",),
            feature_definitions=(_make_feature_def(),),
            target_definition=_make_target_def(horizon=2),
            rows=rows,
        )

        cfg = SplitConfig(
            train_fraction=0.20,
            val_fraction=0.50,
            test_fraction=0.30,
            purge_overlap=True,
            embargo_bars=0,
        )
        result = splitter.split(matrix, cfg)

        # 1. Purged row counts
        assert result.val_rows_purged == 1
        assert result.test_rows_purged == 1
        assert result.total_removed_rows == 2

        # 2. Retained partition sizes
        assert result.train_size == 2
        assert result.val_size == 4
        assert result.test_size == 2
        assert result.retained_rows == 8

        # 3. Exact observation dates retained in each partition
        assert tuple(r.observation_date for r in result.train_rows) == (
            date(2024, 1, 4),
            date(2024, 1, 5),
        )
        assert tuple(r.observation_date for r in result.val_rows) == (
            date(2024, 1, 9),
            date(2024, 1, 10),
            date(2024, 1, 11),
            date(2024, 1, 12),
        )
        assert tuple(r.observation_date for r in result.test_rows) == (
            date(2024, 1, 17),
            date(2024, 1, 18),
        )

        # 4. Boundary metadata reflects final retained splits
        assert result.boundary.last_train_observation_date == date(2024, 1, 5)
        assert result.boundary.first_val_observation_date == date(2024, 1, 9)
        assert result.boundary.last_val_observation_date == date(2024, 1, 12)
        assert result.boundary.first_test_observation_date == date(2024, 1, 17)


# ===========================================================================
# DatasetSplitter - embargo tests
# ===========================================================================


class TestDatasetSplitterEmbargo:
    def test_embargo_removes_rows_after_purge(self, splitter: DatasetSplitter) -> None:
        # horizon=1 (no purge), 20 rows, embargo=2
        # train=14, val_raw=3, test_raw=3
        # val: 0 purged, 2 embargoed -> val_size = 1
        # test: 0 purged, 2 embargoed -> test_size = 1
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            embargo_bars=2,
        )
        result = splitter.split(m, cfg)
        assert result.val_rows_purged == 0
        assert result.val_rows_embargoed == 2
        assert result.val_size == 1
        assert result.test_rows_purged == 0
        assert result.test_rows_embargoed == 2
        assert result.test_size == 1

    def test_embargo_zero_has_no_effect(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            embargo_bars=0,
        )
        result = splitter.split(m, cfg)
        assert result.val_rows_embargoed == 0
        assert result.test_rows_embargoed == 0

    def test_embargo_larger_than_segment_clamps_to_segment_size(
        self, splitter: DatasetSplitter
    ) -> None:
        # embargo=100 > val_size=3, should remove all 3 val rows.
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            embargo_bars=100,
        )
        result = splitter.split(m, cfg)
        assert result.val_rows_embargoed == 3  # clamped to raw val segment size
        assert result.val_size == 0

    def test_embargo_not_double_counted_with_purge(self, splitter: DatasetSplitter) -> None:
        # horizon=3, 30 rows: val_raw=4, 2 purged.
        # Remaining after purge = 2. embargo=1.
        # embargoed=1, val_size=1.
        # Purge + embargo = 2+1 = 3. Total removed = 3. Not 4.
        m = _make_matrix(30, horizon=3)
        cfg = SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            embargo_bars=1,
        )
        result = splitter.split(m, cfg)
        assert result.val_rows_purged == 2
        assert result.val_rows_embargoed == 1
        assert result.val_size == 1
        # Total val accounted for: 2+1+1 = 4 = val_raw_size
        raw_val_size = 4  # floor(0.15*30)
        assert result.val_rows_purged + result.val_rows_embargoed + result.val_size == raw_val_size


# ===========================================================================
# SplitResult tests
# ===========================================================================


class TestSplitResult:
    def test_properties_correct(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.train_size == len(result.train_rows)
        assert result.val_size == len(result.val_rows)
        assert result.test_size == len(result.test_rows)
        assert result.retained_rows == result.train_size + result.val_size + result.test_size
        assert result.total_removed_rows == (
            result.val_rows_purged
            + result.val_rows_embargoed
            + result.test_rows_purged
            + result.test_rows_embargoed
        )

    def test_retained_plus_removed_equals_total(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(30, horizon=3)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.retained_rows + result.total_removed_rows == result.total_rows

    def test_boundary_dates_are_correct(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.boundary.last_train_observation_date == result.train_rows[-1].observation_date
        if result.val_rows:
            assert result.boundary.first_val_observation_date == result.val_rows[0].observation_date
            assert result.boundary.last_val_observation_date == result.val_rows[-1].observation_date
        if result.test_rows:
            assert (
                result.boundary.first_test_observation_date == result.test_rows[0].observation_date
            )

    def test_config_preserved_in_result(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.config is cfg

    def test_result_is_frozen(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        with pytest.raises(FrozenInstanceError):
            result.train_rows = ()  # type: ignore[misc]


# ===========================================================================
# SplitBoundary tests
# ===========================================================================


class TestSplitBoundary:
    def test_boundary_none_dates_when_split_empty(self, splitter: DatasetSplitter) -> None:
        # All val rows purged -> first/last_val_observation_date is None.
        m = _make_matrix(30, horizon=5)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.val_size == 0
        assert result.boundary.first_val_observation_date is None
        assert result.boundary.last_val_observation_date is None

    def test_boundary_dates_present_when_splits_not_empty(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.boundary.first_val_observation_date is not None
        assert result.boundary.last_val_observation_date is not None
        assert result.boundary.first_test_observation_date is not None


# ===========================================================================
# DatasetSplitter - edge case tests
# ===========================================================================


class TestDatasetSplitterEdgeCases:
    def test_minimum_viable_matrix_three_rows(self, splitter: DatasetSplitter) -> None:
        """3 rows is the minimum viable matrix size when using 1/3 fractions."""
        m = _make_matrix(3, horizon=1)
        cfg = SplitConfig(
            train_fraction=1.0 / 3.0,
            val_fraction=1.0 / 3.0,
            test_fraction=1.0 / 3.0,
            purge_overlap=False,
        )
        result = splitter.split(m, cfg)
        assert result.train_size == 1
        assert result.val_size == 1
        assert result.test_size == 1
        assert result.total_rows == 3

    def test_single_row_matrix_rejected_by_validator(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(1)
        cfg = SplitConfig.default()
        with pytest.raises(ValueError):
            splitter.split(m, cfg)

    def test_empty_matrix_rejected_by_validator(self, splitter: DatasetSplitter) -> None:
        m = FeatureMatrix(
            instrument_id=_INST_ID,
            feature_names=("sma_window_5",),
            feature_definitions=(_make_feature_def(),),
            target_definition=_make_target_def(),
            rows=(),
        )
        cfg = SplitConfig.default()
        with pytest.raises(ValueError, match="at least one row"):
            splitter.split(m, cfg)

    def test_purge_and_embargo_both_applied(self, splitter: DatasetSplitter) -> None:
        # horizon=3, 30 rows: val_raw=4, 2 purged, embargo=1, 1 embargoed, val_size=1
        m = _make_matrix(30, horizon=3)
        cfg = SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            purge_overlap=True,
            embargo_bars=1,
        )
        result = splitter.split(m, cfg)
        assert result.val_rows_purged == 2
        assert result.val_rows_embargoed == 1
        assert result.val_size == 1

    def test_no_purge_no_embargo_all_rows_retained(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            purge_overlap=False,
            embargo_bars=0,
        )
        result = splitter.split(m, cfg)
        assert result.total_removed_rows == 0
        assert result.retained_rows == result.total_rows

    def test_large_embargo_empties_both_val_and_test(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(
            train_fraction=0.70,
            val_fraction=0.15,
            test_fraction=0.15,
            purge_overlap=False,
            embargo_bars=999,
        )
        result = splitter.split(m, cfg)
        assert result.val_size == 0
        assert result.test_size == 0
        assert result.val_rows_embargoed == 3  # floor(0.15*20) = 3
        assert result.test_rows_embargoed == 3

    def test_result_row_types_are_datasetrow(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        for row in result.train_rows:
            assert isinstance(row, DatasetRow)
        for row in result.val_rows:
            assert isinstance(row, DatasetRow)
        for row in result.test_rows:
            assert isinstance(row, DatasetRow)

    def test_train_rows_are_first_n_rows_of_matrix(self, splitter: DatasetSplitter) -> None:
        """Training rows must always be exactly the first N rows of the matrix."""
        m = _make_matrix(20, horizon=1)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        expected_train_size = int(0.70 * 20)  # floor
        assert result.train_rows == m.rows[:expected_train_size]

    def test_total_rows_matches_matrix_len(self, splitter: DatasetSplitter) -> None:
        m = _make_matrix(100, horizon=2)
        cfg = SplitConfig(train_fraction=0.70, val_fraction=0.15, test_fraction=0.15)
        result = splitter.split(m, cfg)
        assert result.total_rows == len(m)
