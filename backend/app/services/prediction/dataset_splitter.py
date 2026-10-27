"""Leakage-safe chronological dataset splitter for Phase 5.3."""

from __future__ import annotations

import math
from datetime import date

from app.domain.prediction.dataset import DatasetRow, FeatureMatrix
from app.domain.prediction.split_config import SplitConfig
from app.domain.prediction.split_result import SplitBoundary, SplitResult


class DatasetValidator:
    """Validates a :class:FeatureMatrix against structural and content invariants.

    This is a pure, stateless service: it never accesses repositories, databases,
    or external services, and it never modifies its input.

    The following invariants are checked, in order:

    1. The argument is a :class:FeatureMatrix instance.
    2. At least one row is present.
    3. All rows have the same `instrument_id` as the matrix.
    4. Observation dates are strictly chronologically ordered (no duplicates).
    5. Each row's `realization_date` is strictly after its `observation_date`
       (already guaranteed by :class:DatasetRow, verified here defensively).
    6. Feature names are unique (already guaranteed by :class:FeatureMatrix).
    7. All feature values and target values are finite :class:~decimal.Decimal
       values (already guaranteed by :class:DatasetRow, verified defensively).
    8. Feature and target value are valid (not NaN, not Infinity).

    :meth:alidate raises `ValueError` or `TypeError` on the first
    violation found, with a descriptive message.  It returns `None` on success.
    """

    def validate(self, matrix: FeatureMatrix) -> None:
        if not isinstance(matrix, FeatureMatrix):
            raise TypeError(
                f"DatasetValidator.validate expects a FeatureMatrix, got {type(matrix).__name__}"
            )

        if len(matrix.rows) == 0:
            raise ValueError("FeatureMatrix must contain at least one row")

        # Feature-name uniqueness is enforced by FeatureMatrix.__post_init__,
        # but we assert it here explicitly for defence-in-depth.
        if len(set(matrix.feature_names)) != len(matrix.feature_names):
            raise ValueError(
                f"FeatureMatrix.feature_names contains duplicates; found {matrix.feature_names}"
            )

        prev_obs: date | None = None
        for i, row in enumerate(matrix.rows):
            # Instrument consistency
            if row.instrument_id != matrix.instrument_id:
                raise ValueError(
                    f"FeatureMatrix.rows[{i}] instrument_id ({row.instrument_id}) "
                    f"does not match matrix instrument_id ({matrix.instrument_id})"
                )
            # Chronological ordering
            if prev_obs is not None and row.observation_date <= prev_obs:
                if row.observation_date == prev_obs:
                    raise ValueError(
                        f"FeatureMatrix.rows[{i}] has duplicate observation_date "
                        f"{row.observation_date}"
                    )
                raise ValueError(
                    f"FeatureMatrix rows are not strictly chronologically ordered; "
                    f"row {i} observation_date {row.observation_date} is not after "
                    f"preceding date {prev_obs}"
                )
            # Realization strictly after observation
            if row.realization_date <= row.observation_date:
                raise ValueError(
                    f"FeatureMatrix.rows[{i}] realization_date ({row.realization_date}) "
                    f"must be strictly after observation_date ({row.observation_date})"
                )
            # Finite feature values (defensive re-check)
            for j, fv in enumerate(row.feature_values):
                if not fv.is_finite():
                    raise ValueError(
                        f"FeatureMatrix.rows[{i}].feature_values[{j}] is not finite: {fv}"
                    )
            # Finite target value
            if not row.target_value.is_finite():
                raise ValueError(
                    f"FeatureMatrix.rows[{i}].target_value is not finite: {row.target_value}"
                )
            prev_obs = row.observation_date


class DatasetSplitter:
    """Splits a validated :class:FeatureMatrix chronologically into train / val / test sets.

    This is a pure, stateless service.  It never accesses repositories,
    databases, or external services, and never modifies its inputs.

    Algorithm
    ---------
    Given *n* total rows and a :class:SplitConfig:

    1. **Allocate** raw segment sizes::

           train_size = floor(train_fraction * n)
           val_size   = floor(val_fraction   * n)
           test_size  = n - train_size - val_size

       The remainder from integer rounding is absorbed by the test split.

    2. **Validate** that all raw segments contain at least one row.

    3. **Purge** overlapping target intervals from the head of the validation
       segment (when `config.purge_overlap is True`).  A validation row is
       purged when::

           row.observation_date < last_train_realization_date

       where `last_train_realization_date` is the `realization_date` of the
       last row in the training segment.

    4. **Embargo** the first `config.embargo_bars` rows from the head of the
       validation segment (after purging).

    5. Repeat steps 3 and 4 for the test segment, using the last row of the
       *retained* validation segment (after purging and embargo) as the boundary.
       If the retained validation segment is empty, the boundary is the last
       training row.

    6. Build and return an immutable :class:SplitResult.

    Raises
    ------
    TypeError
        If *matrix* or *config* have incorrect types.
    ValueError
        If the matrix is invalid (via :class:DatasetValidator) or if any
        raw segment would be empty after allocation.
    """

    def __init__(self) -> None:
        self._validator = DatasetValidator()

    @property
    def validator(self) -> DatasetValidator:
        return self._validator

    def split(self, matrix: FeatureMatrix, config: SplitConfig) -> SplitResult:
        """Split *matrix* chronologically according to *config*.

        Parameters
        ----------
        matrix:
            A validated :class:FeatureMatrix.  Must contain at least 3 rows
            (so that each raw segment has at least one row).
        config:
            The :class:SplitConfig controlling fractions, purging, and embargo.

        Returns
        -------
        SplitResult
            An immutable result containing the three data partitions together
            with full audit metadata.

        Raises
        ------
        TypeError
            If *matrix* is not a :class:FeatureMatrix or *config* is not a
            :class:SplitConfig.
        ValueError
            If the matrix fails validation, or if any raw allocation segment
            would have zero rows.
        """
        if not isinstance(config, SplitConfig):
            raise TypeError(f"config must be a SplitConfig, got {type(config).__name__}")

        # Validate matrix structural invariants
        self._validator.validate(matrix)

        rows: tuple[DatasetRow, ...] = matrix.rows
        n = len(rows)

        # --- 1. Allocate raw segment sizes -----------------------------------
        train_size = math.floor(config.train_fraction * n)
        val_size = math.floor(config.val_fraction * n)
        test_size = n - train_size - val_size  # absorbs rounding remainder

        if train_size < 1:
            raise ValueError(
                f"train_fraction {config.train_fraction} allocates 0 rows from {n} total rows; "
                f"increase train_fraction or provide more data"
            )
        if val_size < 1:
            raise ValueError(
                f"val_fraction {config.val_fraction} allocates 0 rows from {n} total rows; "
                f"increase val_fraction or provide more data"
            )
        if test_size < 1:
            raise ValueError(
                f"test_fraction {config.test_fraction} allocates 0 rows from {n} total rows; "
                f"increase test_fraction or provide more data"
            )

        # --- 2. Slice raw segments -------------------------------------------
        train_raw: tuple[DatasetRow, ...] = rows[:train_size]
        val_raw: tuple[DatasetRow, ...] = rows[train_size : train_size + val_size]
        test_raw: tuple[DatasetRow, ...] = rows[train_size + val_size :]

        # --- 3 & 4. Purge and embargo validation split -----------------------
        val_rows, val_purged, val_embargoed = _apply_purge_and_embargo(
            segment=val_raw,
            boundary_row=train_raw[-1],
            purge_overlap=config.purge_overlap,
            embargo_bars=config.embargo_bars,
        )

        # --- 5. Purge and embargo test split ---------------------------------
        # Boundary is the last retained row in val; fall back to last train row
        # when val is completely empty after purging + embargo.
        val_boundary_row = val_rows[-1] if val_rows else train_raw[-1]
        test_rows, test_purged, test_embargoed = _apply_purge_and_embargo(
            segment=test_raw,
            boundary_row=val_boundary_row,
            purge_overlap=config.purge_overlap,
            embargo_bars=config.embargo_bars,
        )

        # --- 6. Build boundary metadata --------------------------------------
        boundary = SplitBoundary(
            last_train_observation_date=train_raw[-1].observation_date,
            first_val_observation_date=val_rows[0].observation_date if val_rows else None,
            last_val_observation_date=val_rows[-1].observation_date if val_rows else None,
            first_test_observation_date=test_rows[0].observation_date if test_rows else None,
        )

        return SplitResult(
            train_rows=train_raw,
            val_rows=val_rows,
            test_rows=test_rows,
            config=config,
            boundary=boundary,
            val_rows_purged=val_purged,
            val_rows_embargoed=val_embargoed,
            test_rows_purged=test_purged,
            test_rows_embargoed=test_embargoed,
            total_rows=n,
        )


def _apply_purge_and_embargo(
    segment: tuple[DatasetRow, ...],
    boundary_row: DatasetRow,
    purge_overlap: bool,
    embargo_bars: int,
) -> tuple[tuple[DatasetRow, ...], int, int]:
    """Apply target-interval purging and embargo to the head of *segment*.

    Parameters
    ----------
    segment:
        Raw chronologically ordered rows to process.
    boundary_row:
        The last *kept* row in the preceding split.  Its `realization_date`
        defines the overlap boundary.
    purge_overlap:
        When `True`, rows whose `observation_date` is strictly less than
        `boundary_row.realization_date` are removed.
    embargo_bars:
        Number of additional rows to remove after purging.

    Returns
    -------
    kept : tuple[DatasetRow, ...]
        Remaining rows after purging and embargo.
    purged_count : int
        Number of rows removed by purging.
    embargoed_count : int
        Number of rows removed by embargo (after purging).
    """
    purged_count = 0
    embargoed_count = 0

    remaining = segment

    if purge_overlap:
        last_realization = boundary_row.realization_date
        cut = 0
        for row in remaining:
            if row.observation_date < last_realization:
                cut += 1
            else:
                break
        purged_count = cut
        remaining = remaining[cut:]

    if embargo_bars > 0 and remaining:
        embargo_cut = min(embargo_bars, len(remaining))
        embargoed_count = embargo_cut
        remaining = remaining[embargo_cut:]

    return remaining, purged_count, embargoed_count
