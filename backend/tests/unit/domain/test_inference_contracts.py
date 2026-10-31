"""Unit tests for Phase 5.5.1 domain inference contracts (InferenceRow, InferenceMatrix)."""

from datetime import date, datetime
from decimal import Decimal
from uuid import uuid4

import pytest

from app.domain.features.feature_category import FeatureCategory
from app.domain.features.feature_definition import FeatureDefinition
from app.domain.prediction.inference_matrix import InferenceMatrix
from app.domain.prediction.inference_row import InferenceRow


def _make_feature_def(name: str = "sma_5", window: int = 5) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        category=FeatureCategory.TECHNICAL,
        description="Simple Moving Average",
        parameters={"window": window},
    )


class TestInferenceRowContract:
    def test_inference_row_valid_creation(self) -> None:
        inst_id = uuid4()
        obs_date = date(2026, 3, 15)
        f_vals = (Decimal("101.50"), Decimal("0.025"), Decimal("-0.010"))

        row = InferenceRow(
            instrument_id=inst_id,
            observation_date=obs_date,
            feature_values=f_vals,
        )

        assert row.instrument_id == inst_id
        assert row.observation_date == obs_date
        assert row.feature_values == f_vals
        # Crucial architectural invariant: unlabeled inference rows have NO target or realization fields
        assert not hasattr(row, "target_value")
        assert not hasattr(row, "realization_date")

    def test_inference_row_to_dict(self) -> None:
        inst_id = uuid4()
        obs_date = date(2026, 3, 15)
        row = InferenceRow(
            instrument_id=inst_id,
            observation_date=obs_date,
            feature_values=(Decimal("10.5"), Decimal("20.5")),
        )

        as_dict = row.to_dict(["sma_5", "sma_20"])
        assert as_dict == {
            "instrument_id": inst_id,
            "observation_date": obs_date,
            "sma_5": Decimal("10.5"),
            "sma_20": Decimal("20.5"),
        }

    def test_inference_row_to_dict_mismatched_names_raises(self) -> None:
        row = InferenceRow(
            instrument_id=uuid4(),
            observation_date=date(2026, 3, 15),
            feature_values=(Decimal("10.5"), Decimal("20.5")),
        )
        with pytest.raises(ValueError, match="feature_names length"):
            row.to_dict(["only_one_name"])

    def test_inference_row_to_dict_rejects_non_sequence(self) -> None:
        row = InferenceRow(
            instrument_id=uuid4(),
            observation_date=date(2026, 3, 15),
            feature_values=(Decimal("10.5"),),
        )
        with pytest.raises(TypeError, match="feature_names must be a sequence"):
            row.to_dict(123)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="feature_names must be a sequence"):
            row.to_dict("single_string")

    def test_inference_row_to_dict_rejects_empty_or_whitespace_name(self) -> None:
        row = InferenceRow(
            instrument_id=uuid4(),
            observation_date=date(2026, 3, 15),
            feature_values=(Decimal("10.5"), Decimal("20.5")),
        )
        with pytest.raises(ValueError, match="must be a non-empty string"):
            row.to_dict(["", "valid_name"])
        with pytest.raises(ValueError, match="must be a non-empty string"):
            row.to_dict(["   ", "valid_name"])
        with pytest.raises(ValueError, match="must be a non-empty string"):
            row.to_dict([None, "valid_name"])  # type: ignore[list-item]

    def test_inference_row_to_dict_rejects_duplicate_names(self) -> None:
        row = InferenceRow(
            instrument_id=uuid4(),
            observation_date=date(2026, 3, 15),
            feature_values=(Decimal("10.5"), Decimal("20.5")),
        )
        with pytest.raises(ValueError, match="feature_names must be unique"):
            row.to_dict(["sma_5", "sma_5"])

    def test_inference_row_to_dict_rejects_reserved_metadata_collisions(self) -> None:
        row = InferenceRow(
            instrument_id=uuid4(),
            observation_date=date(2026, 3, 15),
            feature_values=(Decimal("10.5"), Decimal("20.5")),
        )
        with pytest.raises(ValueError, match="collides with reserved row attribute"):
            row.to_dict(["instrument_id", "feature_2"])
        with pytest.raises(ValueError, match="collides with reserved row attribute"):
            row.to_dict(["feature_1", "observation_date"])

    def test_inference_row_rejects_invalid_instrument_id(self) -> None:
        with pytest.raises(TypeError, match="must be a UUID instance"):
            InferenceRow(
                instrument_id="not-a-uuid",  # type: ignore[arg-type]
                observation_date=date(2026, 3, 15),
                feature_values=(Decimal("1"),),
            )

    def test_inference_row_rejects_datetime_as_observation_date(self) -> None:
        with pytest.raises(TypeError, match="must be a date, not a datetime"):
            InferenceRow(
                instrument_id=uuid4(),
                observation_date=datetime(2026, 3, 15, 9, 30),
                feature_values=(Decimal("1"),),
            )

    def test_inference_row_rejects_non_date_observation_date(self) -> None:
        with pytest.raises(TypeError, match="must be a date instance"):
            InferenceRow(
                instrument_id=uuid4(),
                observation_date="2026-03-15",  # type: ignore[arg-type]
                feature_values=(Decimal("1"),),
            )

    def test_inference_row_rejects_non_tuple_feature_values(self) -> None:
        with pytest.raises(TypeError, match="must be a tuple or list of Decimal"):
            InferenceRow(
                instrument_id=uuid4(),
                observation_date=date(2026, 3, 15),
                feature_values=Decimal("1"),  # type: ignore[arg-type]
            )

    def test_inference_row_rejects_non_decimal_in_feature_values(self) -> None:
        with pytest.raises(TypeError, match="must be a Decimal instance"):
            InferenceRow(
                instrument_id=uuid4(),
                observation_date=date(2026, 3, 15),
                feature_values=(1.23,),  # type: ignore[arg-type]
            )

    def test_inference_row_rejects_non_finite_decimals(self) -> None:
        for non_finite in (Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")):
            with pytest.raises(ValueError, match="must be a finite Decimal"):
                InferenceRow(
                    instrument_id=uuid4(),
                    observation_date=date(2026, 3, 15),
                    feature_values=(non_finite,),
                )

    def test_inference_row_preserves_finite_negative_feature_values(self) -> None:
        row = InferenceRow(
            instrument_id=uuid4(),
            observation_date=date(2026, 3, 15),
            feature_values=(Decimal("-0.0525"), Decimal("-100.5")),
        )
        assert row.feature_values == (Decimal("-0.0525"), Decimal("-100.5"))


class TestInferenceMatrixContract:
    def test_inference_matrix_valid_creation_and_properties(self) -> None:
        inst_id = uuid4()
        f_defs = (_make_feature_def("f1", 5), _make_feature_def("f2", 10))
        f_names = ("f1", "f2")

        r1 = InferenceRow(inst_id, date(2026, 1, 1), (Decimal("1.0"), Decimal("2.0")))
        r2 = InferenceRow(inst_id, date(2026, 1, 2), (Decimal("3.0"), Decimal("4.0")))

        matrix = InferenceMatrix(
            instrument_id=inst_id,
            feature_names=f_names,
            feature_definitions=f_defs,
            rows=(r1, r2),
            dropped_incomplete_feature_count=1,
        )

        assert matrix.instrument_id == inst_id
        assert matrix.feature_names == f_names
        assert matrix.feature_definitions == f_defs
        assert len(matrix) == 2
        assert matrix.dropped_incomplete_feature_count == 1
        assert matrix.feature_matrix == (
            (Decimal("1.0"), Decimal("2.0")),
            (Decimal("3.0"), Decimal("4.0")),
        )
        assert matrix.observation_dates == (date(2026, 1, 1), date(2026, 1, 2))

    def test_inference_matrix_allows_empty_rows(self) -> None:
        inst_id = uuid4()
        matrix = InferenceMatrix(
            instrument_id=inst_id,
            feature_names=("f1",),
            feature_definitions=(_make_feature_def("f1"),),
            rows=(),
        )
        assert len(matrix) == 0
        assert matrix.feature_matrix == ()
        assert matrix.observation_dates == ()

    def test_inference_matrix_rejects_empty_features(self) -> None:
        inst_id = uuid4()
        with pytest.raises(ValueError, match="must contain at least one feature"):
            InferenceMatrix(
                instrument_id=inst_id,
                feature_names=(),
                feature_definitions=(),
                rows=(),
            )

    def test_inference_matrix_rejects_non_uuid_instrument_id(self) -> None:
        with pytest.raises(TypeError, match="must be a UUID instance"):
            InferenceMatrix(
                instrument_id="not-a-uuid",  # type: ignore[arg-type]
                feature_names=("f1",),
                feature_definitions=(_make_feature_def("f1"),),
                rows=(),
            )

    def test_inference_matrix_rejects_duplicate_feature_names(self) -> None:
        with pytest.raises(ValueError, match="must be unique"):
            InferenceMatrix(
                instrument_id=uuid4(),
                feature_names=("f1", "f1"),
                feature_definitions=(_make_feature_def("f1"), _make_feature_def("f1")),
                rows=(),
            )

    def test_inference_matrix_rejects_mismatched_definitions_count(self) -> None:
        with pytest.raises(ValueError, match="must equal feature_definitions length"):
            InferenceMatrix(
                instrument_id=uuid4(),
                feature_names=("f1", "f2"),
                feature_definitions=(_make_feature_def("f1"),),
                rows=(),
            )

    def test_inference_matrix_rejects_row_with_different_instrument_id(self) -> None:
        inst_a = uuid4()
        inst_b = uuid4()
        row = InferenceRow(inst_b, date(2026, 1, 1), (Decimal("1.0"),))
        with pytest.raises(ValueError, match="does not match matrix instrument_id"):
            InferenceMatrix(
                instrument_id=inst_a,
                feature_names=("f1",),
                feature_definitions=(_make_feature_def("f1"),),
                rows=(row,),
            )

    def test_inference_matrix_rejects_row_with_mismatched_feature_width(self) -> None:
        inst_id = uuid4()
        row = InferenceRow(inst_id, date(2026, 1, 1), (Decimal("1.0"), Decimal("2.0")))
        with pytest.raises(ValueError, match="does not match feature count"):
            InferenceMatrix(
                instrument_id=inst_id,
                feature_names=("f1",),  # width 1 vs row width 2
                feature_definitions=(_make_feature_def("f1"),),
                rows=(row,),
            )

    def test_inference_matrix_rejects_duplicate_observation_dates(self) -> None:
        inst_id = uuid4()
        r1 = InferenceRow(inst_id, date(2026, 1, 1), (Decimal("1.0"),))
        r2 = InferenceRow(inst_id, date(2026, 1, 1), (Decimal("2.0"),))  # duplicate!
        with pytest.raises(ValueError, match="duplicate observation_date"):
            InferenceMatrix(
                instrument_id=inst_id,
                feature_names=("f1",),
                feature_definitions=(_make_feature_def("f1"),),
                rows=(r1, r2),
            )

    def test_inference_matrix_rejects_non_chronological_rows(self) -> None:
        inst_id = uuid4()
        r1 = InferenceRow(inst_id, date(2026, 1, 5), (Decimal("1.0"),))
        r2 = InferenceRow(inst_id, date(2026, 1, 2), (Decimal("2.0"),))  # backwards!
        with pytest.raises(ValueError, match="must be chronologically ordered"):
            InferenceMatrix(
                instrument_id=inst_id,
                feature_names=("f1",),
                feature_definitions=(_make_feature_def("f1"),),
                rows=(r1, r2),
            )

    def test_inference_matrix_rejects_invalid_drop_counter(self) -> None:
        inst_id = uuid4()
        # Bool as int
        with pytest.raises(TypeError, match="must be an int"):
            InferenceMatrix(
                instrument_id=inst_id,
                feature_names=("f1",),
                feature_definitions=(_make_feature_def("f1"),),
                rows=(),
                dropped_incomplete_feature_count=True,
            )
        # Negative int
        with pytest.raises(ValueError, match="must be non-negative"):
            InferenceMatrix(
                instrument_id=inst_id,
                feature_names=("f1",),
                feature_definitions=(_make_feature_def("f1"),),
                rows=(),
                dropped_incomplete_feature_count=-1,
            )
