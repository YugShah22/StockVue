"""Unit tests for Phase 5.5.1 ML dataset adapter (ml_dataset_adapter.py)."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID, uuid4

import numpy as np
import pytest

from app.domain.entities.market_bar import MarketBar
from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.prediction.dataset import DatasetRow, FeatureMatrix
from app.domain.prediction.inference_matrix import InferenceMatrix
from app.domain.prediction.inference_row import InferenceRow
from app.domain.prediction.split_config import SplitConfig
from app.domain.prediction.split_result import SplitBoundary, SplitResult
from app.domain.prediction.target_definition import TargetDefinition
from app.services.features.feature_calculator_registry import create_default_feature_registry
from app.services.features.feature_engine import FeatureEngine
from app.services.prediction.dataset_builder import DatasetBuilder, generate_feature_name
from app.services.prediction.dataset_splitter import DatasetSplitter
from app.services.prediction.ml_dataset_adapter import (
    NumpyInferenceDataset,
    NumpySplit,
    _safe_decimal_to_float64,
    to_numpy_inference,
    to_numpy_split,
    verify_schema_alignment,
)

_INST_ID = UUID("00000000-0000-0000-0000-000000000001")


def _make_feature_def(name: str = "sma", window: int = 5) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        category=FeatureCategory.TECHNICAL,
        description="Simple moving average",
        parameters={"window": window},
    )


def _make_target_def(horizon: int = 1) -> TargetDefinition:
    return TargetDefinition(name="forward_return", horizon_bars=horizon)


class TestSafeDecimalToFloat64:
    def test_valid_conversions(self) -> None:
        assert _safe_decimal_to_float64(Decimal("100"), "test") == 100.0
        assert _safe_decimal_to_float64(Decimal("0"), "test") == 0.0
        assert _safe_decimal_to_float64(Decimal("0.000"), "test") == 0.0
        assert _safe_decimal_to_float64(Decimal("-0.0525"), "test") == -0.0525
        assert _safe_decimal_to_float64(Decimal("123456.789"), "test") == 123456.789

    def test_rejects_non_decimal(self) -> None:
        with pytest.raises(TypeError, match="expected Decimal"):
            _safe_decimal_to_float64(100.0, "test")  # type: ignore[arg-type]

    def test_rejects_non_finite_decimal(self) -> None:
        for val in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with pytest.raises(ValueError, match="non-finite Decimal"):
                _safe_decimal_to_float64(val, "test")

    def test_rejects_overflow(self) -> None:
        # 1e400 overflows 64-bit IEEE 754 float
        overflow_dec = Decimal("1e400")
        with pytest.raises(ValueError, match="overflows float64"):
            _safe_decimal_to_float64(overflow_dec, "test")

        overflow_neg = Decimal("-1e400")
        with pytest.raises(ValueError, match="overflows float64"):
            _safe_decimal_to_float64(overflow_neg, "test")

    def test_rejects_underflow_to_zero(self) -> None:
        # Non-zero Decimal that is smaller than minimum positive subnormal float64 (5e-324)
        underflow_dec = Decimal("1e-400")
        with pytest.raises(ValueError, match="underflowed to 0.0"):
            _safe_decimal_to_float64(underflow_dec, "test")

        underflow_neg = Decimal("-1e-400")
        with pytest.raises(ValueError, match="underflowed to 0.0"):
            _safe_decimal_to_float64(underflow_neg, "test")


class TestNumpyContainersReadonlyProtection:
    def test_numpy_split_read_only_protection(self) -> None:
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)

        row_train = DatasetRow(
            inst_id, date(2026, 1, 1), date(2026, 1, 2), (Decimal("10"),), Decimal("0.01")
        )
        row_val = DatasetRow(
            inst_id, date(2026, 1, 2), date(2026, 1, 3), (Decimal("20"),), Decimal("0.02")
        )
        row_test = DatasetRow(
            inst_id, date(2026, 1, 3), date(2026, 1, 4), (Decimal("30"),), Decimal("0.03")
        )

        cfg = SplitConfig(train_fraction=0.34, val_fraction=0.34, test_fraction=0.32)
        b = SplitBoundary(date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 2), date(2026, 1, 3))
        split_res = SplitResult(
            train_rows=(row_train,),
            val_rows=(row_val,),
            test_rows=(row_test,),
            config=cfg,
            boundary=b,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=3,
        )

        matrix = FeatureMatrix(
            instrument_id=inst_id,
            feature_names=(f_name,),
            feature_definitions=(f_def,),
            target_definition=_make_target_def(),
            rows=(row_train, row_val, row_test),
        )

        numpy_split = to_numpy_split(matrix, split_res)

        assert isinstance(numpy_split, NumpySplit)

        # 1. Assert C-contiguous and float64
        assert numpy_split.X_train.flags.c_contiguous
        assert numpy_split.X_train.dtype == np.float64
        assert numpy_split.y_train.dtype == np.float64

        # 2. Assert read-only protection on all arrays
        assert not numpy_split.X_train.flags.writeable
        assert not numpy_split.y_train.flags.writeable
        assert not numpy_split.X_val.flags.writeable
        assert not numpy_split.y_val.flags.writeable
        assert not numpy_split.X_test.flags.writeable
        assert not numpy_split.y_test.flags.writeable

        # 3. Assert in-place mutation attempts raise ValueError
        with pytest.raises(ValueError, match="read-only"):
            numpy_split.X_train[0, 0] = 999.0

        with pytest.raises(ValueError, match="read-only"):
            numpy_split.y_train[0] = 999.0

        with pytest.raises(ValueError, match="read-only"):
            numpy_split.X_val.fill(0.0)

        # 4. Explicit copy allows modification without mutating original
        writable_copy = numpy_split.X_train.copy()
        writable_copy[0, 0] = 999.0
        assert writable_copy[0, 0] == 999.0
        assert numpy_split.X_train[0, 0] == 10.0

    def test_numpy_inference_dataset_read_only_protection(self) -> None:
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)

        row = InferenceRow(inst_id, date(2026, 1, 1), (Decimal("50.0"),))
        matrix = InferenceMatrix(
            instrument_id=inst_id,
            feature_names=(f_name,),
            feature_definitions=(f_def,),
            rows=(row,),
        )

        inf_ds = to_numpy_inference(matrix)

        assert isinstance(inf_ds, NumpyInferenceDataset)
        assert inf_ds.X.flags.c_contiguous
        assert inf_ds.X.dtype == np.float64
        assert not inf_ds.X.flags.writeable
        assert inf_ds.X.shape == (1, 1)
        assert inf_ds.X[0, 0] == 50.0

        with pytest.raises(ValueError, match="read-only"):
            inf_ds.X[0, 0] = 123.0

        writable_copy = inf_ds.X.copy()
        writable_copy[0, 0] = 123.0
        assert inf_ds.X[0, 0] == 50.0


class TestToNumpySplitValidation:
    def _build_valid_fixture(
        self,
    ) -> tuple[FeatureMatrix, SplitResult]:
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)

        r1 = DatasetRow(
            inst_id, date(2026, 1, 1), date(2026, 1, 2), (Decimal("10"),), Decimal("0.01")
        )
        r2 = DatasetRow(
            inst_id, date(2026, 1, 2), date(2026, 1, 3), (Decimal("20"),), Decimal("0.02")
        )
        r3 = DatasetRow(
            inst_id, date(2026, 1, 3), date(2026, 1, 4), (Decimal("30"),), Decimal("0.03")
        )
        r4 = DatasetRow(
            inst_id, date(2026, 1, 4), date(2026, 1, 5), (Decimal("40"),), Decimal("0.04")
        )

        cfg = SplitConfig(train_fraction=0.50, val_fraction=0.25, test_fraction=0.25)
        b = SplitBoundary(date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 3), date(2026, 1, 4))
        split_res = SplitResult(
            train_rows=(r1, r2),
            val_rows=(r3,),
            test_rows=(r4,),
            config=cfg,
            boundary=b,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )

        matrix = FeatureMatrix(
            instrument_id=inst_id,
            feature_names=(f_name,),
            feature_definitions=(f_def,),
            target_definition=_make_target_def(),
            rows=(r1, r2, r3, r4),
        )

        return matrix, split_res

    def test_successful_conversion_and_shapes(self) -> None:
        matrix, split_res = self._build_valid_fixture()
        res = to_numpy_split(matrix, split_res)

        assert res.X_train.shape == (2, 1)
        assert res.y_train.shape == (2,)
        assert res.X_val.shape == (1, 1)
        assert res.y_val.shape == (1,)
        assert res.X_test.shape == (1, 1)
        assert res.y_test.shape == (1,)
        assert np.array_equal(res.X_train, np.array([[10.0], [20.0]]))
        assert np.array_equal(res.y_train, np.array([0.01, 0.02]))
        assert res.feature_names == matrix.feature_names
        assert res.feature_definitions == matrix.feature_definitions
        assert res.split_result == split_res

    def test_rejects_total_rows_mismatch(self) -> None:
        matrix, split_res = self._build_valid_fixture()
        # Create matrix with 3 rows instead of 4
        m_short = FeatureMatrix(
            instrument_id=matrix.instrument_id,
            feature_names=matrix.feature_names,
            feature_definitions=matrix.feature_definitions,
            target_definition=matrix.target_definition,
            rows=matrix.rows[:3],
        )
        with pytest.raises(ValueError, match="total_rows .* does not match"):
            to_numpy_split(m_short, split_res)

    def test_rejects_split_accounting_invariant_violation(self) -> None:
        matrix, split_res = self._build_valid_fixture()
        bad_split = SplitResult(
            train_rows=split_res.train_rows,
            val_rows=split_res.val_rows,
            test_rows=split_res.test_rows,
            config=split_res.config,
            boundary=split_res.boundary,
            val_rows_purged=99,  # Bad accounting: 4 retained + 99 != 4 total
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="SplitResult accounting invariant violated"):
            to_numpy_split(matrix, bad_split)

    def test_rejects_feature_name_mismatch_against_definitions(self) -> None:
        matrix, split_res = self._build_valid_fixture()
        # Matrix with uncanonical name
        m_bad_name = FeatureMatrix(
            instrument_id=matrix.instrument_id,
            feature_names=("wrong_name",),
            feature_definitions=matrix.feature_definitions,
            target_definition=matrix.target_definition,
            rows=matrix.rows,
        )
        with pytest.raises(ValueError, match="does not match expected generated name"):
            to_numpy_split(m_bad_name, split_res)

    def test_rejects_mismatched_instrument_id_in_split(self) -> None:
        matrix, split_res = self._build_valid_fixture()
        other_id = uuid4()
        r_other = DatasetRow(
            other_id, date(2026, 1, 4), date(2026, 1, 5), (Decimal("40"),), Decimal("0.04")
        )
        bad_split = SplitResult(
            train_rows=split_res.train_rows,
            val_rows=split_res.val_rows,
            test_rows=(r_other,),
            config=split_res.config,
            boundary=split_res.boundary,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="instrument_id .* does not match"):
            to_numpy_split(matrix, bad_split)

    def test_rejects_duplicate_dates_across_partitions(self) -> None:
        matrix, split_res = self._build_valid_fixture()
        # Row 2 in both train and val
        bad_split = SplitResult(
            train_rows=split_res.train_rows,
            val_rows=(split_res.train_rows[1],),  # Duplicate of Jan 2!
            test_rows=split_res.test_rows,
            config=split_res.config,
            boundary=split_res.boundary,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="Duplicate observation_date"):
            to_numpy_split(matrix, bad_split)

    def test_rejects_non_chronological_partition_boundaries(self) -> None:
        matrix, split_res = self._build_valid_fixture()
        # Train ends on Jan 3, val starts on Jan 2 -> out of order
        bad_boundary = SplitBoundary(
            date(2026, 1, 3), date(2026, 1, 2), date(2026, 1, 2), date(2026, 1, 4)
        )
        bad_split = SplitResult(
            train_rows=(matrix.rows[0], matrix.rows[2]),  # Jan 1, Jan 3
            val_rows=(matrix.rows[1],),  # Jan 2
            test_rows=(matrix.rows[3],),  # Jan 4
            config=split_res.config,
            boundary=bad_boundary,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="Chronological split boundary violated"):
            to_numpy_split(matrix, bad_split)

    def test_rejects_mismatched_feature_values_against_matrix(self) -> None:
        matrix, split_res = self._build_valid_fixture()
        # Modified feature value on row 1
        tampered_r1 = DatasetRow(
            matrix.instrument_id,
            date(2026, 1, 1),
            date(2026, 1, 2),
            (Decimal("9999.0"),),  # Mismatch! Source has 10
            Decimal("0.01"),
        )
        bad_split = SplitResult(
            train_rows=(tampered_r1, matrix.rows[1]),
            val_rows=split_res.val_rows,
            test_rows=split_res.test_rows,
            config=split_res.config,
            boundary=split_res.boundary,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="feature_values mismatch source row"):
            to_numpy_split(matrix, bad_split)

    def test_rejects_mismatched_target_value_against_matrix(self) -> None:
        matrix, split_res = self._build_valid_fixture()
        tampered_r1 = DatasetRow(
            matrix.instrument_id,
            date(2026, 1, 1),
            date(2026, 1, 2),
            matrix.rows[0].feature_values,
            Decimal("0.9999"),  # Mismatch!
        )
        bad_split = SplitResult(
            train_rows=(tampered_r1, matrix.rows[1]),
            val_rows=split_res.val_rows,
            test_rows=split_res.test_rows,
            config=split_res.config,
            boundary=split_res.boundary,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="target_value .* mismatches source row"):
            to_numpy_split(matrix, bad_split)

    def test_purge_and_embargo_gap_verification(self) -> None:
        """Validates that legitimate exclusions (purged/embargoed rows) are verified accurately."""
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)

        # 6 daily rows: Jan 1 to Jan 6
        dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(6)]
        rows = [
            DatasetRow(
                inst_id,
                d,
                d + timedelta(days=2),
                (Decimal(str(10 * (i + 1))),),
                Decimal("0.01"),
            )
            for i, d in enumerate(dates)
        ]

        # Train: Jan 1, Jan 2
        # Jan 3 is purged (gap between Jan 2 and Jan 4) -> 1 purged
        # Val: Jan 4
        # Jan 5 is purged (gap between Jan 4 and Jan 6) -> 1 purged
        # Test: Jan 6
        b = SplitBoundary(
            last_train_observation_date=date(2026, 1, 2),
            first_val_observation_date=date(2026, 1, 4),
            last_val_observation_date=date(2026, 1, 4),
            first_test_observation_date=date(2026, 1, 6),
        )
        valid_split = SplitResult(
            train_rows=(rows[0], rows[1]),
            val_rows=(rows[3],),
            test_rows=(rows[5],),
            config=SplitConfig(train_fraction=0.34, val_fraction=0.34, test_fraction=0.32),
            boundary=b,
            val_rows_purged=1,
            val_rows_embargoed=0,
            test_rows_purged=1,
            test_rows_embargoed=0,
            total_rows=6,
        )

        matrix = FeatureMatrix(
            instrument_id=inst_id,
            feature_names=(f_name,),
            feature_definitions=(f_def,),
            target_definition=_make_target_def(horizon=2),
            rows=tuple(rows),
        )

        # Should succeed because Jan 3 and Jan 5 gaps match recorded purge counts
        res = to_numpy_split(matrix, valid_split)
        assert res.X_train.shape == (2, 1)
        assert res.X_val.shape == (1, 1)
        assert res.X_test.shape == (1, 1)

        # Now test that an incorrect purge count in SplitResult is rejected
        mismatched_purge_split = SplitResult(
            train_rows=(rows[0], rows[1]),
            val_rows=(rows[3],),
            test_rows=(rows[5],),
            config=SplitConfig(train_fraction=0.34, val_fraction=0.34, test_fraction=0.32),
            boundary=b,
            val_rows_purged=2,  # Claims 2 purged, but only Jan 3 is in gap!
            val_rows_embargoed=0,
            test_rows_purged=0,  # Claims 0 purged
            test_rows_embargoed=0,
            total_rows=6,
        )
        with pytest.raises(ValueError, match="Split purge/embargo mismatch"):
            to_numpy_split(matrix, mismatched_purge_split)


class TestVerifySchemaAlignment:
    def test_schema_alignment_success(self) -> None:
        inst_id = _INST_ID
        f1 = _make_feature_def("sma", 5)
        f2 = _make_feature_def("sma", 20)
        n1 = generate_feature_name(f1)
        n2 = generate_feature_name(f2)

        m_train = FeatureMatrix(
            instrument_id=inst_id,
            feature_names=(n1, n2),
            feature_definitions=(f1, f2),
            target_definition=_make_target_def(),
            rows=(),
        )

        m_inf = InferenceMatrix(
            instrument_id=inst_id,
            feature_names=(n1, n2),
            feature_definitions=(f1, f2),
            rows=(),
        )

        # Passes without error
        verify_schema_alignment(m_train, m_inf)

    def test_schema_alignment_mismatched_column_count(self) -> None:
        inst_id = _INST_ID
        f1 = _make_feature_def("sma", 5)
        f2 = _make_feature_def("sma", 20)
        n1 = generate_feature_name(f1)
        n2 = generate_feature_name(f2)

        m_train = FeatureMatrix(
            instrument_id=inst_id,
            feature_names=(n1, n2),
            feature_definitions=(f1, f2),
            target_definition=_make_target_def(),
            rows=(),
        )

        m_inf = InferenceMatrix(
            instrument_id=inst_id,
            feature_names=(n1,),
            feature_definitions=(f1,),
            rows=(),
        )

        with pytest.raises(ValueError, match="Feature count mismatch"):
            verify_schema_alignment(m_train, m_inf)

    def test_schema_alignment_reordered_columns(self) -> None:
        inst_id = _INST_ID
        f1 = _make_feature_def("sma", 5)
        f2 = _make_feature_def("sma", 20)
        n1 = generate_feature_name(f1)
        n2 = generate_feature_name(f2)

        m_train = FeatureMatrix(
            instrument_id=inst_id,
            feature_names=(n1, n2),
            feature_definitions=(f1, f2),
            target_definition=_make_target_def(),
            rows=(),
        )

        # Reordered
        m_inf = InferenceMatrix(
            instrument_id=inst_id,
            feature_names=(n2, n1),
            feature_definitions=(f2, f1),
            rows=(),
        )

        with pytest.raises(ValueError, match="Feature column mismatch"):
            verify_schema_alignment(m_train, m_inf)


class TestNumpyContainersValidation:
    """Tests direct validation invariants in NumpySplit and NumpyInferenceDataset."""

    def _make_valid_split_kwargs(self) -> dict[str, Any]:
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)
        r_tr = DatasetRow(
            inst_id, date(2026, 1, 1), date(2026, 1, 2), (Decimal("10"),), Decimal("0.01")
        )
        r_va = DatasetRow(
            inst_id, date(2026, 1, 2), date(2026, 1, 3), (Decimal("20"),), Decimal("0.02")
        )
        r_te = DatasetRow(
            inst_id, date(2026, 1, 3), date(2026, 1, 4), (Decimal("30"),), Decimal("0.03")
        )
        cfg = SplitConfig(train_fraction=0.34, val_fraction=0.34, test_fraction=0.32)
        b = SplitBoundary(date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 2), date(2026, 1, 3))
        split_res = SplitResult(
            train_rows=(r_tr,),
            val_rows=(r_va,),
            test_rows=(r_te,),
            config=cfg,
            boundary=b,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=3,
        )

        def _make_arr(val: float, is_2d: bool = True) -> np.ndarray:
            if is_2d:
                a = np.array([[val]], dtype=np.float64, order="C")
            else:
                a = np.array([val], dtype=np.float64, order="C")
            a.flags.writeable = False
            return a

        return {
            "X_train": _make_arr(10.0, is_2d=True),
            "y_train": _make_arr(0.01, is_2d=False),
            "X_val": _make_arr(20.0, is_2d=True),
            "y_val": _make_arr(0.02, is_2d=False),
            "X_test": _make_arr(30.0, is_2d=True),
            "y_test": _make_arr(0.03, is_2d=False),
            "feature_names": (f_name,),
            "feature_definitions": (f_def,),
            "split_result": split_res,
        }

    def _create_split(self, **overrides: Any) -> NumpySplit:
        kwargs = self._make_valid_split_kwargs()
        kwargs.update(overrides)
        return NumpySplit(
            X_train=kwargs["X_train"],
            y_train=kwargs["y_train"],
            X_val=kwargs["X_val"],
            y_val=kwargs["y_val"],
            X_test=kwargs["X_test"],
            y_test=kwargs["y_test"],
            feature_names=kwargs["feature_names"],
            feature_definitions=kwargs["feature_definitions"],
            split_result=kwargs["split_result"],
        )

    def test_numpy_split_rejects_non_finite_values(self) -> None:
        for non_finite in (np.nan, np.inf, -np.inf):
            bad_arr = np.array([[non_finite]], dtype=np.float64, order="C")
            bad_arr.flags.writeable = False
            with pytest.raises(ValueError, match="contains non-finite values"):
                self._create_split(X_train=bad_arr)

            bad_y = np.array([non_finite], dtype=np.float64, order="C")
            bad_y.flags.writeable = False
            with pytest.raises(ValueError, match="contains non-finite values"):
                self._create_split(y_train=bad_y)

    def test_numpy_split_rejects_non_owning_views(self) -> None:
        base_arr = np.zeros((2, 1), dtype=np.float64, order="C")
        slice_view = base_arr[1:]
        slice_view.flags.writeable = False
        assert slice_view.base is not None
        with pytest.raises(ValueError, match="must own its memory buffer"):
            self._create_split(X_train=slice_view)

    def test_numpy_split_rejects_non_float64(self) -> None:
        bad_arr = np.ones((1, 1), dtype=np.float32, order="C")
        bad_arr.flags.writeable = False
        with pytest.raises(ValueError, match="dtype must be float64"):
            self._create_split(X_train=bad_arr)

    def test_numpy_split_rejects_non_c_contiguous(self) -> None:
        # 2D Fortran contiguous array (transposed 2D array)
        f_arr = np.zeros((2, 1), dtype=np.float64, order="C").T
        f_arr.flags.writeable = False
        with pytest.raises((ValueError, TypeError)):
            self._create_split(X_train=f_arr)

    def test_numpy_split_rejects_writeable_arrays(self) -> None:
        writeable_arr = np.ones((1, 1), dtype=np.float64, order="C")
        with pytest.raises(ValueError, match="must have writeable=False"):
            self._create_split(X_train=writeable_arr)

    def test_numpy_split_validates_feature_names_and_definitions(self) -> None:
        with pytest.raises(ValueError, match="must be a non-empty string"):
            self._create_split(feature_names=("",))
        with pytest.raises(ValueError, match="must be a non-empty string"):
            self._create_split(feature_names=("   ",))
        with pytest.raises(ValueError, match="must be unique"):
            f_def = _make_feature_def("sma", 5)
            self._create_split(feature_names=("sma", "sma"), feature_definitions=(f_def, f_def))
        with pytest.raises(TypeError, match="must be a FeatureDefinition"):
            self._create_split(feature_definitions=("not_a_def",))
        with pytest.raises(ValueError, match="length .* must equal"):
            self._create_split(feature_definitions=())

    def test_numpy_split_rejects_empty_features(self) -> None:
        with pytest.raises(ValueError, match="must contain at least one feature"):
            self._create_split(feature_names=(), feature_definitions=())

    def test_numpy_split_rejects_mismatched_target_value(self) -> None:
        """Approach A: Array target values must match source domain rows."""
        bad_y = np.array([999.0], dtype=np.float64, order="C")
        bad_y.flags.writeable = False
        with pytest.raises(ValueError, match="target at index 0 .* does not match source row"):
            self._create_split(y_train=bad_y)

    def test_numpy_split_rejects_mismatched_feature_value(self) -> None:
        """Approach A: Array feature values must match source domain rows."""
        bad_x = np.array([[999.0]], dtype=np.float64, order="C")
        bad_x.flags.writeable = False
        with pytest.raises(ValueError, match="feature at \\[0, 0\\] .* does not match source row"):
            self._create_split(X_train=bad_x)

    def _create_inference_dataset(self, **overrides: Any) -> NumpyInferenceDataset:
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)
        r = InferenceRow(inst_id, date(2026, 1, 1), (Decimal("10"),))
        inf_matrix = InferenceMatrix(inst_id, (f_name,), (f_def,), (r,))
        arr = np.array([[10.0]], dtype=np.float64, order="C")
        arr.flags.writeable = False
        kwargs: dict[str, Any] = {
            "X": arr,
            "feature_names": (f_name,),
            "feature_definitions": (f_def,),
            "source_matrix": inf_matrix,
        }
        kwargs.update(overrides)
        return NumpyInferenceDataset(
            X=kwargs["X"],
            feature_names=kwargs["feature_names"],
            feature_definitions=kwargs["feature_definitions"],
            source_matrix=kwargs["source_matrix"],
        )

    def test_numpy_inference_dataset_rejects_non_finite_and_views(self) -> None:
        # Non-finite
        for non_finite in (np.nan, np.inf, -np.inf):
            bad_arr = np.array([[non_finite]], dtype=np.float64, order="C")
            bad_arr.flags.writeable = False
            with pytest.raises(ValueError, match="contains non-finite values"):
                self._create_inference_dataset(X=bad_arr)

        # Non-owning view
        base = np.zeros((2, 1), dtype=np.float64, order="C")
        view = base[1:]
        view.flags.writeable = False
        with pytest.raises(ValueError, match="must own its memory buffer"):
            self._create_inference_dataset(X=view)

        # Non-float64
        float32_arr = np.ones((1, 1), dtype=np.float32, order="C")
        float32_arr.flags.writeable = False
        with pytest.raises(ValueError, match="dtype must be float64"):
            self._create_inference_dataset(X=float32_arr)

        # Writeable
        w_arr = np.ones((1, 1), dtype=np.float64, order="C")
        with pytest.raises(ValueError, match="must have writeable=False"):
            self._create_inference_dataset(X=w_arr)

    def test_numpy_inference_dataset_rejects_empty_features(self) -> None:
        with pytest.raises(ValueError, match="must contain at least one feature"):
            self._create_inference_dataset(feature_names=(), feature_definitions=())

    def test_numpy_inference_dataset_rejects_mismatched_feature_value(self) -> None:
        """Approach A: Array feature values must match source domain matrix."""
        bad_x = np.array([[999.0]], dtype=np.float64, order="C")
        bad_x.flags.writeable = False
        with pytest.raises(ValueError, match="feature at \\[0, 0\\] .* does not match source row"):
            self._create_inference_dataset(X=bad_x)

    def test_numpy_inference_dataset_rejects_mismatched_feature_names(self) -> None:
        """Reject feature_names that differ from source_matrix.feature_names."""
        with pytest.raises(
            ValueError,
            match="NumpyInferenceDataset.feature_names must match source_matrix.feature_names",
        ):
            self._create_inference_dataset(feature_names=("different_name",))

    def test_numpy_inference_dataset_rejects_mismatched_feature_definitions(self) -> None:
        """Reject feature_definitions that differ from source_matrix.feature_definitions."""
        other_def = _make_feature_def("sma", 10)
        with pytest.raises(
            ValueError,
            match="NumpyInferenceDataset.feature_definitions must match source_matrix.feature_definitions",
        ):
            self._create_inference_dataset(feature_definitions=(other_def,))


class TestToNumpySplitBoundaryAndExclusionSemantics:
    """Tests boundary verification, empty partition semantics, and purge/embargo enforcement."""

    def test_rejects_boundary_metadata_mismatch_train(self) -> None:
        matrix, split_res = TestToNumpySplitValidation()._build_valid_fixture()
        bad_b = SplitBoundary(
            last_train_observation_date=date(2026, 1, 1),  # Actual last train is Jan 2
            first_val_observation_date=split_res.boundary.first_val_observation_date,
            last_val_observation_date=split_res.boundary.last_val_observation_date,
            first_test_observation_date=split_res.boundary.first_test_observation_date,
        )
        bad_split = SplitResult(
            train_rows=split_res.train_rows,
            val_rows=split_res.val_rows,
            test_rows=split_res.test_rows,
            config=split_res.config,
            boundary=bad_b,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="Boundary last_train_observation_date"):
            to_numpy_split(matrix, bad_split)

    def test_rejects_boundary_metadata_mismatch_val(self) -> None:
        matrix, split_res = TestToNumpySplitValidation()._build_valid_fixture()
        bad_b = SplitBoundary(
            last_train_observation_date=split_res.boundary.last_train_observation_date,
            first_val_observation_date=date(2026, 1, 9),  # Mismatched!
            last_val_observation_date=split_res.boundary.last_val_observation_date,
            first_test_observation_date=split_res.boundary.first_test_observation_date,
        )
        bad_split = SplitResult(
            train_rows=split_res.train_rows,
            val_rows=split_res.val_rows,
            test_rows=split_res.test_rows,
            config=split_res.config,
            boundary=bad_b,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="Boundary first_val_observation_date"):
            to_numpy_split(matrix, bad_split)

    def test_rejects_boundary_metadata_mismatch_test(self) -> None:
        matrix, split_res = TestToNumpySplitValidation()._build_valid_fixture()
        bad_b = SplitBoundary(
            last_train_observation_date=split_res.boundary.last_train_observation_date,
            first_val_observation_date=split_res.boundary.first_val_observation_date,
            last_val_observation_date=split_res.boundary.last_val_observation_date,
            first_test_observation_date=date(2026, 1, 9),  # Mismatched!
        )
        bad_split = SplitResult(
            train_rows=split_res.train_rows,
            val_rows=split_res.val_rows,
            test_rows=split_res.test_rows,
            config=split_res.config,
            boundary=bad_b,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="Boundary first_test_observation_date"):
            to_numpy_split(matrix, bad_split)

    def test_empty_validation_partition_is_rejected(self) -> None:
        """When validation partition has all rows purged/embargoed, to_numpy_split rejects it."""
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)

        # 4 rows: train gets 2, val gets 1, test gets 1
        r1 = DatasetRow(
            inst_id, date(2026, 1, 1), date(2026, 1, 3), (Decimal("10"),), Decimal("0.01")
        )
        r2 = DatasetRow(
            inst_id, date(2026, 1, 2), date(2026, 1, 4), (Decimal("20"),), Decimal("0.02")
        )
        r3 = DatasetRow(
            inst_id, date(2026, 1, 3), date(2026, 1, 5), (Decimal("30"),), Decimal("0.03")
        )
        r4 = DatasetRow(
            inst_id, date(2026, 1, 4), date(2026, 1, 6), (Decimal("40"),), Decimal("0.04")
        )

        cfg = SplitConfig(
            train_fraction=0.50, val_fraction=0.25, test_fraction=0.25, purge_overlap=True
        )
        b = SplitBoundary(
            last_train_observation_date=date(2026, 1, 2),
            first_val_observation_date=None,
            last_val_observation_date=None,
            first_test_observation_date=date(2026, 1, 4),
        )
        split_res = SplitResult(
            train_rows=(r1, r2),
            val_rows=(),  # empty!
            test_rows=(r4,),
            config=cfg,
            boundary=b,
            val_rows_purged=1,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        matrix = FeatureMatrix(inst_id, (f_name,), (f_def,), _make_target_def(2), (r1, r2, r3, r4))
        with pytest.raises(ValueError, match="split_result.val_rows cannot be empty"):
            to_numpy_split(matrix, split_res)

    def test_empty_test_partition_is_rejected(self) -> None:
        """When test partition has all rows purged/embargoed, to_numpy_split rejects it."""
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)

        # 4 rows: train=2, val=1, test=1
        r1 = DatasetRow(
            inst_id, date(2026, 1, 1), date(2026, 1, 3), (Decimal("10"),), Decimal("0.01")
        )
        r2 = DatasetRow(
            inst_id, date(2026, 1, 2), date(2026, 1, 4), (Decimal("20"),), Decimal("0.02")
        )
        r3 = DatasetRow(
            inst_id, date(2026, 1, 4), date(2026, 1, 6), (Decimal("30"),), Decimal("0.03")
        )
        r4 = DatasetRow(
            inst_id, date(2026, 1, 5), date(2026, 1, 7), (Decimal("40"),), Decimal("0.04")
        )

        cfg = SplitConfig(
            train_fraction=0.50, val_fraction=0.25, test_fraction=0.25, purge_overlap=True
        )
        b = SplitBoundary(
            last_train_observation_date=date(2026, 1, 2),
            first_val_observation_date=date(2026, 1, 4),
            last_val_observation_date=date(2026, 1, 4),
            first_test_observation_date=None,
        )
        split_res = SplitResult(
            train_rows=(r1, r2),
            val_rows=(r3,),
            test_rows=(),  # empty!
            config=cfg,
            boundary=b,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=1,
            test_rows_embargoed=0,
            total_rows=4,
        )
        matrix = FeatureMatrix(inst_id, (f_name,), (f_def,), _make_target_def(2), (r1, r2, r3, r4))
        with pytest.raises(ValueError, match="split_result.test_rows cannot be empty"):
            to_numpy_split(matrix, split_res)

    def test_rejects_missing_intermediate_observation_in_partition(self) -> None:
        """Reject splits where a row is omitted from the middle of a partition."""
        matrix, split_res = TestToNumpySplitValidation()._build_valid_fixture()
        # Train expected 2 rows: (r1, r2). If caller passed only (r1,), accounting catches it
        bad_split = SplitResult(
            train_rows=(matrix.rows[0],),  # Dropped r2!
            val_rows=split_res.val_rows,
            test_rows=split_res.test_rows,
            config=split_res.config,
            boundary=SplitBoundary(
                matrix.rows[0].observation_date,
                split_res.boundary.first_val_observation_date,
                split_res.boundary.last_val_observation_date,
                split_res.boundary.first_test_observation_date,
            ),
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="SplitResult accounting invariant violated"):
            to_numpy_split(matrix, bad_split)

    def test_rejects_tampered_middle_observation_substituted_in_train(self) -> None:
        """Reject splits where an intermediate row in train is substituted with an unrelated row."""
        matrix, split_res = TestToNumpySplitValidation()._build_valid_fixture()
        # Tamper r2 (middle observation in train)
        tampered_r2 = DatasetRow(
            matrix.instrument_id,
            date(2026, 1, 2),
            date(2026, 1, 3),
            (Decimal("7777"),),  # Mismatched features!
            Decimal("0.02"),
        )
        bad_split = SplitResult(
            train_rows=(matrix.rows[0], tampered_r2),
            val_rows=split_res.val_rows,
            test_rows=split_res.test_rows,
            config=split_res.config,
            boundary=split_res.boundary,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=4,
        )
        with pytest.raises(ValueError, match="Train row 1 feature_values mismatch source row"):
            to_numpy_split(matrix, bad_split)

    def test_rejects_missing_tail_observation_in_test_partition(self) -> None:
        """Reject splits where a row is dropped from test partition without legitimate purge/embargo."""
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)
        # 5 rows: train=2, val=1, test=2
        rows = [
            DatasetRow(
                inst_id,
                date(2026, 1, i + 1),
                date(2026, 1, i + 2),
                (Decimal("10"),),
                Decimal("0.01"),
            )
            for i in range(5)
        ]
        cfg = SplitConfig(train_fraction=0.40, val_fraction=0.20, test_fraction=0.40)
        b = SplitBoundary(date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 3), date(2026, 1, 4))
        # Caller attempts to drop rows[4] from test and hide it by claiming it was purged
        bad_split = SplitResult(
            train_rows=(rows[0], rows[1]),
            val_rows=(rows[2],),
            test_rows=(rows[3],),  # Dropped rows[4]!
            config=cfg,
            boundary=b,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=1,  # Falsely claims tail row was purged!
            test_rows_embargoed=0,
            total_rows=5,
        )
        matrix = FeatureMatrix(inst_id, (f_name,), (f_def,), _make_target_def(), tuple(rows))
        with pytest.raises(ValueError, match="Split purge/embargo mismatch for test"):
            to_numpy_split(matrix, bad_split)

    def test_purge_boundary_equality_validation_not_purged(self) -> None:
        """Exact equality at the purge boundary: row.observation_date == max_prior_realization is NOT purged."""
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)

        # 5 rows:
        # r1: Jan 1 -> real Jan 2
        # r2: Jan 2 -> real Jan 3 (max train realization = Jan 3)
        # r3 (val[0]): obs Jan 3 (EQUAL to max_train_realization!) -> real Jan 4
        # r4 (val[1]): obs Jan 4 -> real Jan 5
        # r5 (test[0]): obs Jan 5 -> real Jan 6
        r1 = DatasetRow(
            inst_id, date(2026, 1, 1), date(2026, 1, 2), (Decimal("10"),), Decimal("0.01")
        )
        r2 = DatasetRow(
            inst_id, date(2026, 1, 2), date(2026, 1, 3), (Decimal("20"),), Decimal("0.02")
        )
        r3 = DatasetRow(
            inst_id, date(2026, 1, 3), date(2026, 1, 4), (Decimal("30"),), Decimal("0.03")
        )
        r4 = DatasetRow(
            inst_id, date(2026, 1, 4), date(2026, 1, 5), (Decimal("40"),), Decimal("0.04")
        )
        r5 = DatasetRow(
            inst_id, date(2026, 1, 5), date(2026, 1, 6), (Decimal("50"),), Decimal("0.05")
        )

        cfg = SplitConfig(
            train_fraction=0.40, val_fraction=0.40, test_fraction=0.20, purge_overlap=True
        )
        b = SplitBoundary(
            last_train_observation_date=date(2026, 1, 2),
            first_val_observation_date=date(2026, 1, 3),
            last_val_observation_date=date(2026, 1, 4),
            first_test_observation_date=date(2026, 1, 5),
        )
        # Because obs Jan 3 == real Jan 3, target interval (obs, real] does not overlap -> val_rows_purged = 0
        split_res = SplitResult(
            train_rows=(r1, r2),
            val_rows=(r3, r4),
            test_rows=(r5,),
            config=cfg,
            boundary=b,
            val_rows_purged=0,  # 0 purged! Equality boundary is retained
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=5,
        )
        matrix = FeatureMatrix(
            inst_id, (f_name,), (f_def,), _make_target_def(1), (r1, r2, r3, r4, r5)
        )
        res = to_numpy_split(matrix, split_res)
        assert res.X_val.shape == (2, 1)
        assert res.y_val.shape == (2,)

        # Falsely claiming this boundary row was purged must be rejected
        bad_split = SplitResult(
            train_rows=(r1, r2),
            val_rows=(r4,),
            test_rows=(r5,),
            config=cfg,
            boundary=SplitBoundary(
                date(2026, 1, 2), date(2026, 1, 4), date(2026, 1, 4), date(2026, 1, 5)
            ),
            val_rows_purged=1,  # Falsely claims Jan 3 was purged!
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=5,
        )
        with pytest.raises(ValueError, match="Split purge/embargo mismatch for val"):
            to_numpy_split(matrix, bad_split)

    def test_purge_boundary_equality_test_not_purged(self) -> None:
        """Exact equality at the test purge boundary is NOT purged."""
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)

        # 5 rows:
        # r1: Jan 1 -> real Jan 2
        # r2: Jan 2 -> real Jan 3
        # r3 (val[0]): obs Jan 3 -> real Jan 4 (max_prior_realization = Jan 4)
        # r4 (test[0]): obs Jan 4 (EQUAL to max_prior_realization!) -> real Jan 5
        # r5 (test[1]): obs Jan 5 -> real Jan 6
        r1 = DatasetRow(
            inst_id, date(2026, 1, 1), date(2026, 1, 2), (Decimal("10"),), Decimal("0.01")
        )
        r2 = DatasetRow(
            inst_id, date(2026, 1, 2), date(2026, 1, 3), (Decimal("20"),), Decimal("0.02")
        )
        r3 = DatasetRow(
            inst_id, date(2026, 1, 3), date(2026, 1, 4), (Decimal("30"),), Decimal("0.03")
        )
        r4 = DatasetRow(
            inst_id, date(2026, 1, 4), date(2026, 1, 5), (Decimal("40"),), Decimal("0.04")
        )
        r5 = DatasetRow(
            inst_id, date(2026, 1, 5), date(2026, 1, 6), (Decimal("50"),), Decimal("0.05")
        )

        cfg = SplitConfig(
            train_fraction=0.40, val_fraction=0.20, test_fraction=0.40, purge_overlap=True
        )
        b = SplitBoundary(
            last_train_observation_date=date(2026, 1, 2),
            first_val_observation_date=date(2026, 1, 3),
            last_val_observation_date=date(2026, 1, 3),
            first_test_observation_date=date(2026, 1, 4),
        )
        # Equality: test row is NOT purged
        split_res = SplitResult(
            train_rows=(r1, r2),
            val_rows=(r3,),
            test_rows=(r4, r5),
            config=cfg,
            boundary=b,
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=0,
            test_rows_embargoed=0,
            total_rows=5,
        )
        matrix = FeatureMatrix(
            inst_id, (f_name,), (f_def,), _make_target_def(1), (r1, r2, r3, r4, r5)
        )
        res = to_numpy_split(matrix, split_res)
        assert res.X_test.shape == (2, 1)
        assert res.y_test.shape == (2,)

        # Falsely claiming test was purged is rejected
        bad_split = SplitResult(
            train_rows=(r1, r2),
            val_rows=(r3,),
            test_rows=(r5,),
            config=cfg,
            boundary=SplitBoundary(
                date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 3), date(2026, 1, 5)
            ),
            val_rows_purged=0,
            val_rows_embargoed=0,
            test_rows_purged=1,
            test_rows_embargoed=0,
            total_rows=5,
        )
        with pytest.raises(ValueError, match="Split purge/embargo mismatch for test"):
            to_numpy_split(matrix, bad_split)

    def test_purging_test_partition_when_validation_is_entirely_empty_is_rejected(self) -> None:
        """When validation partition is entirely empty due to purging, to_numpy_split rejects it."""
        inst_id = _INST_ID
        f_def = _make_feature_def("sma", 5)
        f_name = generate_feature_name(f_def)

        # Distinct observation dates:
        # r1: Jan 1 (real Jan 4)
        # r2: Jan 2 (real Jan 5) -> max train real = Jan 5
        # r3 (val): Jan 3 (real Jan 6) -> Jan 3 < Jan 5 -> PURGED!
        # r4 (test[0]): Jan 4 (real Jan 7) -> Jan 4 < Jan 5 -> PURGED!
        # r5 (test[1]): Jan 5 (real Jan 8) -> Jan 5 == Jan 5 -> KEPT!
        r1 = DatasetRow(
            inst_id, date(2026, 1, 1), date(2026, 1, 4), (Decimal("10"),), Decimal("0.01")
        )
        r2 = DatasetRow(
            inst_id, date(2026, 1, 2), date(2026, 1, 5), (Decimal("20"),), Decimal("0.02")
        )
        r3 = DatasetRow(
            inst_id, date(2026, 1, 3), date(2026, 1, 6), (Decimal("30"),), Decimal("0.03")
        )
        r4 = DatasetRow(
            inst_id, date(2026, 1, 4), date(2026, 1, 7), (Decimal("40"),), Decimal("0.04")
        )
        r5 = DatasetRow(
            inst_id, date(2026, 1, 5), date(2026, 1, 8), (Decimal("50"),), Decimal("0.05")
        )

        cfg = SplitConfig(
            train_fraction=0.40, val_fraction=0.20, test_fraction=0.40, purge_overlap=True
        )
        b = SplitBoundary(
            last_train_observation_date=date(2026, 1, 2),
            first_val_observation_date=None,
            last_val_observation_date=None,
            first_test_observation_date=date(2026, 1, 5),
        )
        split_res = SplitResult(
            train_rows=(r1, r2),
            val_rows=(),  # entirely empty!
            test_rows=(r5,),  # r4 purged, r5 kept
            config=cfg,
            boundary=b,
            val_rows_purged=1,
            val_rows_embargoed=0,
            test_rows_purged=1,
            test_rows_embargoed=0,
            total_rows=5,
        )
        matrix = FeatureMatrix(
            inst_id, (f_name,), (f_def,), _make_target_def(3), (r1, r2, r3, r4, r5)
        )
        with pytest.raises(ValueError, match="split_result.val_rows cannot be empty"):
            to_numpy_split(matrix, split_res)


class TestDatasetSplitterAdapterIntegration:
    """Integration test connecting FeatureMatrix, DatasetSplitter, and to_numpy_split."""

    def test_end_to_end_splitter_and_adapter_pipeline(self) -> None:
        """Build real FeatureMatrix, split via DatasetSplitter, and adapt to NumPy arrays."""
        inst_id = uuid4()
        engine = FeatureEngine(create_default_feature_registry())
        builder = DatasetBuilder(feature_engine=engine)

        # 12 daily bars with predictable prices
        base_time = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
        bars = [
            MarketBar(
                instrument_id=inst_id,
                timestamp=base_time + timedelta(days=i),
                open=Decimal(str(100 + i * 2)),
                high=Decimal(str(105 + i * 2)),
                low=Decimal(str(95 + i * 2)),
                close=Decimal(str(100 + i * 2)),
                volume=Decimal("10000"),
            )
            for i in range(12)
        ]

        features = [
            FeatureDefinition(
                name="sma",
                category=FeatureCategory.TECHNICAL,
                description="2-period SMA",
                parameters={"window": 2},
            ),
            FeatureDefinition(
                name="return",
                category=FeatureCategory.TECHNICAL,
                description="1-period Return",
                parameters={"window": 1},
            ),
        ]
        target = TargetDefinition(name="forward_return", horizon_bars=1)

        # 1. Build real FeatureMatrix (10 valid rows)
        matrix = builder.build_dataset(
            instrument_id=inst_id,
            bars=bars,
            features=features,
            target=target,
        )
        assert len(matrix) == 10
        assert len(matrix.feature_names) == 2

        # 2. Split with actual DatasetSplitter using valid SplitConfig
        config = SplitConfig(
            train_fraction=0.50,
            val_fraction=0.25,
            test_fraction=0.25,
            purge_overlap=True,
            embargo_bars=1,
        )
        splitter = DatasetSplitter()
        split_result = splitter.split(matrix, config)

        assert len(split_result.train_rows) == 5
        assert len(split_result.val_rows) == 1
        assert len(split_result.test_rows) == 2
        assert split_result.val_rows_purged == 0
        assert split_result.val_rows_embargoed == 1
        assert split_result.test_rows_purged == 0
        assert split_result.test_rows_embargoed == 1

        # 3. Pass returned SplitResult to to_numpy_split()
        numpy_split = to_numpy_split(matrix, split_result)

        # 4. Verify all three NumPy partitions have the expected shapes and values
        assert isinstance(numpy_split, NumpySplit)

        assert numpy_split.X_train.shape == (5, 2)
        assert numpy_split.y_train.shape == (5,)
        assert numpy_split.X_val.shape == (1, 2)
        assert numpy_split.y_val.shape == (1,)
        assert numpy_split.X_test.shape == (2, 2)
        assert numpy_split.y_test.shape == (2,)

        for arr in (
            numpy_split.X_train,
            numpy_split.y_train,
            numpy_split.X_val,
            numpy_split.y_val,
            numpy_split.X_test,
            numpy_split.y_test,
        ):
            assert arr.dtype == np.float64
            assert arr.flags.c_contiguous
            assert arr.base is None
            assert not arr.flags.writeable

        # Numerical correspondence with source rows
        for (x_arr, y_arr), split_rows in (
            ((numpy_split.X_train, numpy_split.y_train), split_result.train_rows),
            ((numpy_split.X_val, numpy_split.y_val), split_result.val_rows),
            ((numpy_split.X_test, numpy_split.y_test), split_result.test_rows),
        ):
            for i, r in enumerate(split_rows):
                assert y_arr[i] == float(r.target_value)
                for j, fv in enumerate(r.feature_values):
                    assert x_arr[i, j] == float(fv)

        assert numpy_split.feature_names == matrix.feature_names
        assert numpy_split.feature_definitions == matrix.feature_definitions
        assert numpy_split.split_result == split_result
