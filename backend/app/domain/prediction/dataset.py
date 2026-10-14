from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

from app.domain.features.feature_definition import FeatureDefinition
from app.domain.prediction.target_definition import TargetDefinition


@dataclass(frozen=True)
class DatasetRow:
    """A single point-in-time training/evaluation observation row.

    Attributes:
        instrument_id: UUID of the instrument.
        observation_date: Calendar date of the observation bar (P_t).
        realization_date: Calendar date of the future horizon bar (P_{t+h}).
        feature_values: Tuple of computed feature values as of observation_date.
        target_value: Fractional forward return realised at realization_date.
    """

    instrument_id: UUID
    observation_date: date
    realization_date: date
    feature_values: tuple[Decimal, ...]
    target_value: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_id, UUID):
            raise TypeError(
                f"DatasetRow.instrument_id must be a UUID instance, got {type(self.instrument_id).__name__}"
            )

        if isinstance(self.observation_date, datetime):
            raise TypeError("DatasetRow.observation_date must be a date, not a datetime")
        if not isinstance(self.observation_date, date):
            raise TypeError(
                f"DatasetRow.observation_date must be a date instance, got {type(self.observation_date).__name__}"
            )

        if isinstance(self.realization_date, datetime):
            raise TypeError("DatasetRow.realization_date must be a date, not a datetime")
        if not isinstance(self.realization_date, date):
            raise TypeError(
                f"DatasetRow.realization_date must be a date instance, got {type(self.realization_date).__name__}"
            )

        if self.realization_date <= self.observation_date:
            raise ValueError(
                f"DatasetRow.realization_date ({self.realization_date}) must be "
                f"strictly after observation_date ({self.observation_date})"
            )

        if not isinstance(self.feature_values, (tuple, list)):
            raise TypeError(
                f"DatasetRow.feature_values must be a tuple or list of Decimal, got {type(self.feature_values).__name__}"
            )
        f_tuple = tuple(self.feature_values)
        for i, val in enumerate(f_tuple):
            if not isinstance(val, Decimal):
                raise TypeError(
                    f"DatasetRow.feature_values[{i}] must be a Decimal instance, got {type(val).__name__}"
                )
            if not val.is_finite():
                raise ValueError(
                    f"DatasetRow.feature_values[{i}] must be a finite Decimal, got {val}"
                )
        object.__setattr__(self, "feature_values", f_tuple)

        if not isinstance(self.target_value, Decimal):
            raise TypeError(
                f"DatasetRow.target_value must be a Decimal instance, got {type(self.target_value).__name__}"
            )
        if not self.target_value.is_finite():
            raise ValueError(
                f"DatasetRow.target_value must be a finite Decimal, got {self.target_value}"
            )

    def to_dict(self, feature_names: Sequence[str]) -> dict[str, Any]:
        """Convert row to a name-keyed dictionary for serialization or inspection."""

        if len(feature_names) != len(self.feature_values):
            raise ValueError(
                f"feature_names length ({len(feature_names)}) does not match "
                f"feature_values length ({len(self.feature_values)})"
            )
        record: dict[str, Any] = {
            "instrument_id": self.instrument_id,
            "observation_date": self.observation_date,
            "realization_date": self.realization_date,
            "target": self.target_value,
        }
        for name, val in zip(feature_names, self.feature_values, strict=True):
            record[name] = val
        return record


@dataclass(frozen=True)
class FeatureMatrix:
    """Validated, immutable container for a complete point-in-time tabular dataset.

    All rows have complete feature and target values (no missing or NaN values).
    """

    instrument_id: UUID
    feature_names: tuple[str, ...]
    feature_definitions: tuple[FeatureDefinition, ...]
    target_definition: TargetDefinition
    rows: tuple[DatasetRow, ...]
    dropped_incomplete_feature_count: int = 0
    dropped_incomplete_target_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_id, UUID):
            raise TypeError(
                f"FeatureMatrix.instrument_id must be a UUID instance, got {type(self.instrument_id).__name__}"
            )

        if not isinstance(self.feature_names, (tuple, list)):
            raise TypeError(
                f"FeatureMatrix.feature_names must be a tuple or list of str, got {type(self.feature_names).__name__}"
            )
        f_names = tuple(self.feature_names)
        for i, name in enumerate(f_names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"FeatureMatrix.feature_names[{i}] must be a non-empty string, got {name!r}"
                )
        if len(set(f_names)) != len(f_names):
            raise ValueError(
                f"FeatureMatrix.feature_names must be unique; found duplicates in {f_names}"
            )
        object.__setattr__(self, "feature_names", f_names)

        if not isinstance(self.feature_definitions, (tuple, list)):
            raise TypeError(
                f"FeatureMatrix.feature_definitions must be a tuple or list of FeatureDefinition, got {type(self.feature_definitions).__name__}"
            )
        f_defs = tuple(self.feature_definitions)
        for i, d in enumerate(f_defs):
            if not isinstance(d, FeatureDefinition):
                raise TypeError(
                    f"FeatureMatrix.feature_definitions[{i}] must be a FeatureDefinition, got {type(d).__name__}"
                )
        if len(f_names) != len(f_defs):
            raise ValueError(
                f"FeatureMatrix.feature_names length ({len(f_names)}) must equal "
                f"feature_definitions length ({len(f_defs)})"
            )
        object.__setattr__(self, "feature_definitions", f_defs)

        if not isinstance(self.target_definition, TargetDefinition):
            raise TypeError(
                f"FeatureMatrix.target_definition must be a TargetDefinition, got {type(self.target_definition).__name__}"
            )

        # Drop counters validation (reject booleans as ints)
        if isinstance(self.dropped_incomplete_feature_count, bool) or not isinstance(
            self.dropped_incomplete_feature_count, int
        ):
            raise TypeError(
                f"FeatureMatrix.dropped_incomplete_feature_count must be an int, got {type(self.dropped_incomplete_feature_count).__name__}"
            )
        if self.dropped_incomplete_feature_count < 0:
            raise ValueError(
                f"FeatureMatrix.dropped_incomplete_feature_count must be non-negative, got {self.dropped_incomplete_feature_count}"
            )

        if isinstance(self.dropped_incomplete_target_count, bool) or not isinstance(
            self.dropped_incomplete_target_count, int
        ):
            raise TypeError(
                f"FeatureMatrix.dropped_incomplete_target_count must be an int, got {type(self.dropped_incomplete_target_count).__name__}"
            )
        if self.dropped_incomplete_target_count < 0:
            raise ValueError(
                f"FeatureMatrix.dropped_incomplete_target_count must be non-negative, got {self.dropped_incomplete_target_count}"
            )

        if not isinstance(self.rows, (tuple, list)):
            raise TypeError(
                f"FeatureMatrix.rows must be a tuple or list of DatasetRow, got {type(self.rows).__name__}"
            )
        r_tuples = tuple(self.rows)
        prev_date: date | None = None
        for i, row in enumerate(r_tuples):
            if not isinstance(row, DatasetRow):
                raise TypeError(
                    f"FeatureMatrix.rows[{i}] must be a DatasetRow, got {type(row).__name__}"
                )
            if row.instrument_id != self.instrument_id:
                raise ValueError(
                    f"FeatureMatrix.rows[{i}] instrument_id ({row.instrument_id}) "
                    f"does not match matrix instrument_id ({self.instrument_id})"
                )
            if len(row.feature_values) != len(f_names):
                raise ValueError(
                    f"FeatureMatrix.rows[{i}] feature_values width ({len(row.feature_values)}) "
                    f"does not match feature count ({len(f_names)})"
                )
            if prev_date is not None and row.observation_date <= prev_date:
                if row.observation_date == prev_date:
                    raise ValueError(
                        f"FeatureMatrix.rows[{i}] has duplicate observation_date: {row.observation_date}"
                    )
                raise ValueError(
                    f"FeatureMatrix rows must be chronologically ordered; "
                    f"row {i} ({row.observation_date}) is not after ({prev_date})"
                )
            prev_date = row.observation_date
        object.__setattr__(self, "rows", r_tuples)

    @property
    def feature_matrix(self) -> tuple[tuple[Decimal, ...], ...]:
        """2D tuple of shape (n_samples, n_features)."""
        return tuple(r.feature_values for r in self.rows)

    @property
    def target_vector(self) -> tuple[Decimal, ...]:
        """1D tuple of shape (n_samples,)."""
        return tuple(r.target_value for r in self.rows)

    @property
    def observation_dates(self) -> tuple[date, ...]:
        """1D tuple of observation dates for all complete rows."""
        return tuple(r.observation_date for r in self.rows)

    @property
    def realization_dates(self) -> tuple[date, ...]:
        """1D tuple of target realization dates for all complete rows."""
        return tuple(r.realization_date for r in self.rows)

    def __len__(self) -> int:
        return len(self.rows)
