from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from uuid import UUID

from app.domain.features.feature_definition import FeatureDefinition
from app.domain.prediction.inference_row import InferenceRow


@dataclass(frozen=True)
class InferenceMatrix:
    """Validated, immutable container for chronological unlabeled inference observations.

    All rows have complete feature values (no missing or non-finite values)
    and are strictly ordered chronologically by observation_date for a single instrument.
    """

    instrument_id: UUID
    feature_names: tuple[str, ...]
    feature_definitions: tuple[FeatureDefinition, ...]
    rows: tuple[InferenceRow, ...]
    dropped_incomplete_feature_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.instrument_id, UUID):
            raise TypeError(
                f"InferenceMatrix.instrument_id must be a UUID instance, got {type(self.instrument_id).__name__}"
            )

        if not isinstance(self.feature_names, (tuple, list)):
            raise TypeError(
                f"InferenceMatrix.feature_names must be a tuple or list of str, got {type(self.feature_names).__name__}"
            )
        f_names = tuple(self.feature_names)
        for i, name in enumerate(f_names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"InferenceMatrix.feature_names[{i}] must be a non-empty string, got {name!r}"
                )
        if len(set(f_names)) != len(f_names):
            raise ValueError(
                f"InferenceMatrix.feature_names must be unique; found duplicates in {f_names}"
            )
        if len(f_names) == 0:
            raise ValueError("InferenceMatrix must contain at least one feature")
        object.__setattr__(self, "feature_names", f_names)

        if not isinstance(self.feature_definitions, (tuple, list)):
            raise TypeError(
                f"InferenceMatrix.feature_definitions must be a tuple or list of FeatureDefinition, got {type(self.feature_definitions).__name__}"
            )
        f_defs = tuple(self.feature_definitions)
        for i, d in enumerate(f_defs):
            if not isinstance(d, FeatureDefinition):
                raise TypeError(
                    f"InferenceMatrix.feature_definitions[{i}] must be a FeatureDefinition, got {type(d).__name__}"
                )
        if len(f_names) != len(f_defs):
            raise ValueError(
                f"InferenceMatrix.feature_names length ({len(f_names)}) must equal "
                f"feature_definitions length ({len(f_defs)})"
            )
        object.__setattr__(self, "feature_definitions", f_defs)

        if isinstance(self.dropped_incomplete_feature_count, bool) or not isinstance(
            self.dropped_incomplete_feature_count, int
        ):
            raise TypeError(
                f"InferenceMatrix.dropped_incomplete_feature_count must be an int, got {type(self.dropped_incomplete_feature_count).__name__}"
            )
        if self.dropped_incomplete_feature_count < 0:
            raise ValueError(
                f"InferenceMatrix.dropped_incomplete_feature_count must be non-negative, got {self.dropped_incomplete_feature_count}"
            )

        if not isinstance(self.rows, (tuple, list)):
            raise TypeError(
                f"InferenceMatrix.rows must be a tuple or list of InferenceRow, got {type(self.rows).__name__}"
            )
        r_tuples = tuple(self.rows)
        prev_date: date | None = None
        for i, row in enumerate(r_tuples):
            if not isinstance(row, InferenceRow):
                raise TypeError(
                    f"InferenceMatrix.rows[{i}] must be an InferenceRow, got {type(row).__name__}"
                )
            if row.instrument_id != self.instrument_id:
                raise ValueError(
                    f"InferenceMatrix.rows[{i}] instrument_id ({row.instrument_id}) "
                    f"does not match matrix instrument_id ({self.instrument_id})"
                )
            if len(row.feature_values) != len(f_names):
                raise ValueError(
                    f"InferenceMatrix.rows[{i}] feature_values width ({len(row.feature_values)}) "
                    f"does not match feature count ({len(f_names)})"
                )
            if prev_date is not None and row.observation_date <= prev_date:
                if row.observation_date == prev_date:
                    raise ValueError(
                        f"InferenceMatrix.rows[{i}] has duplicate observation_date: {row.observation_date}"
                    )
                raise ValueError(
                    f"InferenceMatrix rows must be chronologically ordered; "
                    f"row {i} ({row.observation_date}) is not after ({prev_date})"
                )
            prev_date = row.observation_date
        object.__setattr__(self, "rows", r_tuples)

    @property
    def feature_matrix(self) -> tuple[tuple[Decimal, ...], ...]:
        """2D tuple of shape (n_samples, n_features)."""
        return tuple(r.feature_values for r in self.rows)

    @property
    def observation_dates(self) -> tuple[date, ...]:
        """1D tuple of observation dates for all complete rows."""
        return tuple(r.observation_date for r in self.rows)

    def __len__(self) -> int:
        return len(self.rows)
