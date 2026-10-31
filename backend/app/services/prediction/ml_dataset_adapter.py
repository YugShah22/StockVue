import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

import numpy as np

from app.domain.features.feature_definition import FeatureDefinition
from app.domain.prediction.dataset import DatasetRow, FeatureMatrix
from app.domain.prediction.inference_matrix import InferenceMatrix
from app.domain.prediction.split_result import SplitResult
from app.services.prediction.dataset_builder import generate_feature_name


def _safe_decimal_to_float64(val: Decimal, context: str) -> float:
    """Convert a Decimal to float64, rejecting non-finite values, overflows, and underflows to zero.

    Transitioning from Decimal to float64 is an intentional approximation required by ML operations.
    This converter never silently clips, clamps, or replaces invalid numbers.
    """
    if not isinstance(val, Decimal):
        raise TypeError(f"{context}: expected Decimal, got {type(val).__name__}")
    if not val.is_finite():
        raise ValueError(
            f"{context}: non-finite Decimal value {val} cannot be converted to float64"
        )

    try:
        f = float(val)
    except OverflowError as err:
        raise ValueError(
            f"{context}: Decimal value {val} overflows float64 representation"
        ) from err

    if math.isinf(f):
        raise ValueError(f"{context}: Decimal value {val} overflows float64 representation ({f})")

    if not math.isfinite(f):
        raise ValueError(f"{context}: Decimal value {val} produced non-finite float64 {f}")

    if val != 0 and f == 0.0:
        raise ValueError(
            f"{context}: Non-zero Decimal value {val} underflowed to 0.0 in float64 representation"
        )

    return f


@dataclass(frozen=True)
class NumpySplit:
    """Read-only numerical representation of a train/validation/test split.

    Arrays are C-contiguous, float64, and flagged read-only (writeable=False).
    Detailed partition rows, dates, and ID metadata are referenced via split_result.

    Note on Immutability:
        In NumPy, disabling the writeable flag (writeable=False) provides read-only view
        semantics that prevent accidental in-place assignment or modification. Callers
        can technically re-enable write flags if they hold array references; therefore,
        containers own their memory buffers and require downstream consumers that need
        to mutate data to explicitly allocate a working buffer via arr.copy().

    Numerical Consistency (Approach A):
        Container validation strictly verifies that array values match the underlying
        Decimal values in split_result rows via float64 conversion, preventing directly
        constructed containers from holding arbitrary arrays that diverge from source
        domain rows and targets.
    """

    X_train: np.ndarray  # shape (N_train, D), order="C", writeable=False
    y_train: np.ndarray  # shape (N_train,), order="C", writeable=False
    X_val: np.ndarray  # shape (N_val, D), order="C", writeable=False
    y_val: np.ndarray  # shape (N_val,), order="C", writeable=False
    X_test: np.ndarray  # shape (N_test, D), order="C", writeable=False
    y_test: np.ndarray  # shape (N_test,), order="C", writeable=False
    feature_names: tuple[str, ...]
    feature_definitions: tuple[FeatureDefinition, ...]
    split_result: SplitResult

    def __post_init__(self) -> None:
        if not isinstance(self.split_result, SplitResult):
            raise TypeError(
                f"NumpySplit.split_result must be a SplitResult, got {type(self.split_result).__name__}"
            )

        if not isinstance(self.feature_names, (tuple, list)):
            raise TypeError(
                f"NumpySplit.feature_names must be a tuple or list of str, got {type(self.feature_names).__name__}"
            )
        f_names = tuple(self.feature_names)
        for i, name in enumerate(f_names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"NumpySplit.feature_names[{i}] must be a non-empty string, got {name!r}"
                )
        if len(set(f_names)) != len(f_names):
            raise ValueError(
                f"NumpySplit.feature_names must be unique; found duplicates in {f_names}"
            )
        if len(f_names) == 0:
            raise ValueError("NumpySplit must contain at least one feature")
        object.__setattr__(self, "feature_names", f_names)

        if not isinstance(self.feature_definitions, (tuple, list)):
            raise TypeError(
                f"NumpySplit.feature_definitions must be a tuple or list of FeatureDefinition, got {type(self.feature_definitions).__name__}"
            )
        f_defs = tuple(self.feature_definitions)
        for i, d in enumerate(f_defs):
            if not isinstance(d, FeatureDefinition):
                raise TypeError(
                    f"NumpySplit.feature_definitions[{i}] must be a FeatureDefinition, got {type(d).__name__}"
                )
        if len(f_names) != len(f_defs):
            raise ValueError(
                f"NumpySplit.feature_names length ({len(f_names)}) must equal "
                f"feature_definitions length ({len(f_defs)})"
            )
        object.__setattr__(self, "feature_definitions", f_defs)

        num_features = len(f_names)

        for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test"):
            arr = getattr(self, name)
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"NumpySplit.{name} must be a numpy ndarray, got {type(arr).__name__}"
                )
            if arr.dtype != np.float64:
                raise ValueError(f"NumpySplit.{name} dtype must be float64, got {arr.dtype}")
            if not arr.flags.c_contiguous:
                raise ValueError(f"NumpySplit.{name} must be C-contiguous")
            if arr.base is not None:
                raise ValueError(
                    f"NumpySplit.{name} must own its memory buffer (base is not None; non-owning views rejected)"
                )
            if arr.flags.writeable:
                raise ValueError(f"NumpySplit.{name} array must have writeable=False")
            if not bool(np.all(np.isfinite(arr))):
                raise ValueError(
                    f"NumpySplit.{name} contains non-finite values (NaN, +inf, or -inf)"
                )

        # Validate array dimensions and shape consistency
        for x_name, y_name, expected_rows in (
            ("X_train", "y_train", len(self.split_result.train_rows)),
            ("X_val", "y_val", len(self.split_result.val_rows)),
            ("X_test", "y_test", len(self.split_result.test_rows)),
        ):
            x_arr = getattr(self, x_name)
            y_arr = getattr(self, y_name)

            if x_arr.ndim != 2:
                raise ValueError(f"NumpySplit.{x_name} must be a 2D array, got {x_arr.ndim}D")
            if y_arr.ndim != 1:
                raise ValueError(f"NumpySplit.{y_name} must be a 1D array, got {y_arr.ndim}D")

            if x_arr.shape[0] != expected_rows:
                raise ValueError(
                    f"NumpySplit.{x_name} row count ({x_arr.shape[0]}) does not match "
                    f"split row count ({expected_rows})"
                )
            if y_arr.shape[0] != expected_rows:
                raise ValueError(
                    f"NumpySplit.{y_name} row count ({y_arr.shape[0]}) does not match "
                    f"split row count ({expected_rows})"
                )
            if x_arr.shape[1] != num_features:
                raise ValueError(
                    f"NumpySplit.{x_name} feature width ({x_arr.shape[1]}) does not match "
                    f"feature count ({num_features})"
                )

        # Validate numerical content consistency against source domain rows (Approach A)
        for x_arr, y_arr, rows, split_label in (
            (self.X_train, self.y_train, self.split_result.train_rows, "train"),
            (self.X_val, self.y_val, self.split_result.val_rows, "val"),
            (self.X_test, self.y_test, self.split_result.test_rows, "test"),
        ):
            for i, row in enumerate(rows):
                expected_y = _safe_decimal_to_float64(
                    row.target_value, f"NumpySplit y_{split_label} [{i}]"
                )
                if y_arr[i] != expected_y:
                    raise ValueError(
                        f"NumpySplit y_{split_label} target at index {i} ({y_arr[i]}) does not match "
                        f"source row target value ({row.target_value} -> {expected_y})"
                    )
                for j, fv in enumerate(row.feature_values):
                    expected_x = _safe_decimal_to_float64(
                        fv, f"NumpySplit X_{split_label} [{i}, {j}]"
                    )
                    if x_arr[i, j] != expected_x:
                        raise ValueError(
                            f"NumpySplit X_{split_label} feature at [{i}, {j}] ({x_arr[i, j]}) does not match "
                            f"source row feature value ({fv} -> {expected_x})"
                        )


@dataclass(frozen=True)
class NumpyInferenceDataset:
    """Read-only numerical representation of an unlabeled inference matrix.

    Contains exclusively features and schema metadata. Target arrays are strictly omitted.

    Note on Immutability:
        Disabling the writeable flag (writeable=False) enforces read-only view semantics.
        The container owns its contiguous buffer; downstream mutation requires arr.copy().

    Numerical Consistency (Approach A):
        Container validation strictly verifies that X array values match the underlying
        Decimal feature values in source_matrix rows via float64 conversion, preventing
        directly constructed containers from holding arbitrary arrays that diverge from
        source domain rows.
    """

    X: np.ndarray  # shape (N, D), order="C", writeable=False
    feature_names: tuple[str, ...]
    feature_definitions: tuple[FeatureDefinition, ...]
    source_matrix: InferenceMatrix

    def __post_init__(self) -> None:
        if not isinstance(self.source_matrix, InferenceMatrix):
            raise TypeError(
                f"NumpyInferenceDataset.source_matrix must be an InferenceMatrix, got {type(self.source_matrix).__name__}"
            )

        if not isinstance(self.feature_names, (tuple, list)):
            raise TypeError(
                f"NumpyInferenceDataset.feature_names must be a tuple or list of str, got {type(self.feature_names).__name__}"
            )
        f_names = tuple(self.feature_names)
        for i, name in enumerate(f_names):
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"NumpyInferenceDataset.feature_names[{i}] must be a non-empty string, got {name!r}"
                )
        if len(set(f_names)) != len(f_names):
            raise ValueError(
                f"NumpyInferenceDataset.feature_names must be unique; found duplicates in {f_names}"
            )
        if len(f_names) == 0:
            raise ValueError("NumpyInferenceDataset must contain at least one feature")
        object.__setattr__(self, "feature_names", f_names)

        if not isinstance(self.feature_definitions, (tuple, list)):
            raise TypeError(
                f"NumpyInferenceDataset.feature_definitions must be a tuple or list of FeatureDefinition, got {type(self.feature_definitions).__name__}"
            )
        f_defs = tuple(self.feature_definitions)
        for i, d in enumerate(f_defs):
            if not isinstance(d, FeatureDefinition):
                raise TypeError(
                    f"NumpyInferenceDataset.feature_definitions[{i}] must be a FeatureDefinition, got {type(d).__name__}"
                )
        if len(f_names) != len(f_defs):
            raise ValueError(
                f"NumpyInferenceDataset.feature_names length ({len(f_names)}) must equal "
                f"feature_definitions length ({len(f_defs)})"
            )
        object.__setattr__(self, "feature_definitions", f_defs)

        if f_names != self.source_matrix.feature_names:
            raise ValueError(
                "NumpyInferenceDataset.feature_names must match source_matrix.feature_names"
            )
        if f_defs != self.source_matrix.feature_definitions:
            raise ValueError(
                "NumpyInferenceDataset.feature_definitions must match "
                "source_matrix.feature_definitions"
            )

        num_features = len(f_names)

        if not isinstance(self.X, np.ndarray):
            raise TypeError(
                f"NumpyInferenceDataset.X must be a numpy ndarray, got {type(self.X).__name__}"
            )
        if self.X.dtype != np.float64:
            raise ValueError(f"NumpyInferenceDataset.X dtype must be float64, got {self.X.dtype}")
        if not self.X.flags.c_contiguous:
            raise ValueError("NumpyInferenceDataset.X must be C-contiguous")
        if self.X.base is not None:
            raise ValueError(
                "NumpyInferenceDataset.X must own its memory buffer (base is not None; non-owning views rejected)"
            )
        if self.X.flags.writeable:
            raise ValueError("NumpyInferenceDataset.X must have writeable=False")
        if not bool(np.all(np.isfinite(self.X))):
            raise ValueError(
                "NumpyInferenceDataset.X contains non-finite values (NaN, +inf, or -inf)"
            )

        if self.X.ndim != 2:
            raise ValueError(f"NumpyInferenceDataset.X must be a 2D array, got {self.X.ndim}D")
        if self.X.shape[0] != len(self.source_matrix.rows):
            raise ValueError(
                f"NumpyInferenceDataset.X row count ({self.X.shape[0]}) does not match "
                f"source_matrix row count ({len(self.source_matrix.rows)})"
            )
        if self.X.shape[1] != num_features:
            raise ValueError(
                f"NumpyInferenceDataset.X feature width ({self.X.shape[1]}) does not match "
                f"feature count ({num_features})"
            )

        # Validate numerical content consistency against source domain matrix (Approach A)
        for i, row in enumerate(self.source_matrix.rows):
            for j, fv in enumerate(row.feature_values):
                expected_x = _safe_decimal_to_float64(fv, f"NumpyInferenceDataset X [{i}, {j}]")
                if self.X[i, j] != expected_x:
                    raise ValueError(
                        f"NumpyInferenceDataset X feature at [{i}, {j}] ({self.X[i, j]}) does not match "
                        f"source row feature value ({fv} -> {expected_x})"
                    )


def _rows_to_numpy_arrays(
    rows: Sequence[DatasetRow],
    num_features: int,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert DatasetRows to C-contiguous, read-only X (N, D) and y (N,) float64 arrays."""
    n_rows = len(rows)
    if n_rows == 0:
        X = np.empty((0, num_features), dtype=np.float64, order="C")
        y = np.empty((0,), dtype=np.float64, order="C")
        X.flags.writeable = False
        y.flags.writeable = False
        return X, y

    X_buf = np.empty((n_rows, num_features), dtype=np.float64, order="C")
    y_buf = np.empty((n_rows,), dtype=np.float64, order="C")

    for i, row in enumerate(rows):
        if len(row.feature_values) != num_features:
            raise ValueError(
                f"{split_name} row {i} width ({len(row.feature_values)}) does not match expected {num_features}"
            )
        for j, fv in enumerate(row.feature_values):
            X_buf[i, j] = _safe_decimal_to_float64(fv, f"{split_name} feature [{i}, {j}]")
        y_buf[i] = _safe_decimal_to_float64(row.target_value, f"{split_name} target [{i}]")

    X_buf.flags.writeable = False
    y_buf.flags.writeable = False
    return X_buf, y_buf


def _verify_partition_rows_match(
    partition_name: str,
    split_rows: Sequence[DatasetRow],
    expected_rows: Sequence[DatasetRow],
) -> None:
    if len(split_rows) != len(expected_rows):
        raise ValueError(
            f"Retained {partition_name} rows count ({len(split_rows)}) does not match "
            f"expected count ({len(expected_rows)})"
        )
    for idx, (sp_row, exp_row) in enumerate(zip(split_rows, expected_rows, strict=True)):
        if sp_row.instrument_id != exp_row.instrument_id:
            raise ValueError(
                f"{partition_name} row {idx} instrument_id ({sp_row.instrument_id}) does not match "
                f"matrix instrument_id ({exp_row.instrument_id})"
            )
        if sp_row.observation_date != exp_row.observation_date:
            raise ValueError(
                f"{partition_name} row {idx} observation_date ({sp_row.observation_date}) does not match "
                f"source row ({exp_row.observation_date})"
            )
        if sp_row.realization_date != exp_row.realization_date:
            raise ValueError(
                f"{partition_name} row {idx} realization_date ({sp_row.realization_date}) mismatches "
                f"source row ({exp_row.realization_date})"
            )
        if sp_row.target_value != exp_row.target_value:
            raise ValueError(
                f"{partition_name} row {idx} target_value ({sp_row.target_value}) mismatches "
                f"source row ({exp_row.target_value})"
            )
        if sp_row.feature_values != exp_row.feature_values:
            raise ValueError(f"{partition_name} row {idx} feature_values mismatch source row")


def to_numpy_split(
    matrix: FeatureMatrix,
    split_result: SplitResult,
) -> NumpySplit:
    """Convert a FeatureMatrix and SplitResult into read-only float64 arrays.

    Validates that:
    1. split_result.total_rows equals len(matrix.rows).
    2. Retained rows across all splits plus removed rows equals split_result.total_rows.
    3. Every retained split row belongs to matrix.instrument_id and maps to a source row
       in matrix with identical observation_date, realization_date, target_value,
       and feature_values.
    4. Observation dates within each partition are strictly chronological, and partitions
       are mutually disjoint with zero date overlap.
    5. Validation and test partitions must be non-empty (matching DatasetSplitter invariants).
    6. Boundary metadata strictly matches actual retained partition observation dates.
    7. Raw partition allocations from SplitConfig match source FeatureMatrix ordering.
    8. All row exclusions correspond strictly to verified purge and embargo operations
       at the head of validation and test segments. Any row missing from the source
       matrix outside legitimate purge and embargo regions is rejected.
    9. matrix.feature_names matches generate_feature_name for all feature_definitions.
    """
    if not isinstance(matrix, FeatureMatrix):
        raise TypeError(f"matrix must be a FeatureMatrix, got {type(matrix).__name__}")
    if not isinstance(split_result, SplitResult):
        raise TypeError(f"split_result must be a SplitResult, got {type(split_result).__name__}")

    n = len(matrix.rows)
    if split_result.total_rows != n:
        raise ValueError(
            f"split_result.total_rows ({split_result.total_rows}) does not match "
            f"matrix.rows count ({n})"
        )

    if split_result.retained_rows + split_result.total_removed_rows != split_result.total_rows:
        raise ValueError(
            f"SplitResult accounting invariant violated: retained ({split_result.retained_rows}) + "
            f"removed ({split_result.total_removed_rows}) != total ({split_result.total_rows})"
        )

    # 1. Verify schema names against definitions
    if len(matrix.feature_names) != len(matrix.feature_definitions):
        raise ValueError(
            f"FeatureMatrix feature_names length ({len(matrix.feature_names)}) does not match "
            f"feature_definitions length ({len(matrix.feature_definitions)})"
        )
    for idx, (name, defn) in enumerate(
        zip(matrix.feature_names, matrix.feature_definitions, strict=True)
    ):
        expected_name = generate_feature_name(defn)
        if name != expected_name:
            raise ValueError(
                f"FeatureMatrix feature_name at index {idx} ('{name}') does not match "
                f"expected generated name ('{expected_name}')"
            )

    # 2. Verify source FeatureMatrix chronological ordering
    prev_mat_d: date | None = None
    for idx, r in enumerate(matrix.rows):
        if prev_mat_d is not None and r.observation_date <= prev_mat_d:
            raise ValueError(
                f"FeatureMatrix rows are not strictly chronological: row {idx} date "
                f"{r.observation_date} <= {prev_mat_d}"
            )
        prev_mat_d = r.observation_date

    # 3. Validate disjoint partition observation dates
    train_dates = [r.observation_date for r in split_result.train_rows]
    val_dates = [r.observation_date for r in split_result.val_rows]
    test_dates = [r.observation_date for r in split_result.test_rows]

    seen_split_dates: set[date] = set()
    for partition_name, p_dates in (
        ("train", train_dates),
        ("val", val_dates),
        ("test", test_dates),
    ):
        prev_d: date | None = None
        for d in p_dates:
            if prev_d is not None and d <= prev_d:
                raise ValueError(
                    f"Observation dates in {partition_name} split must be strictly increasing: "
                    f"{d} <= {prev_d}"
                )
            if d in seen_split_dates:
                raise ValueError(
                    f"Duplicate observation_date {d} found across split partitions in {partition_name}"
                )
            seen_split_dates.add(d)
            prev_d = d

    # 4. Validate chronological partition boundaries
    if train_dates and val_dates and max(train_dates) >= min(val_dates):
        raise ValueError(
            f"Chronological split boundary violated: max train date ({max(train_dates)}) "
            f">= min val date ({min(val_dates)})"
        )
    if val_dates and test_dates and max(val_dates) >= min(test_dates):
        raise ValueError(
            f"Chronological split boundary violated: max val date ({max(val_dates)}) "
            f">= min test date ({min(test_dates)})"
        )
    if train_dates and test_dates and max(train_dates) >= min(test_dates):
        raise ValueError(
            f"Chronological split boundary violated: max train date ({max(train_dates)}) "
            f">= min test date ({min(test_dates)})"
        )

    # 5. Validate boundary metadata against actual retained partitions
    b = split_result.boundary
    if len(split_result.train_rows) == 0:
        raise ValueError("split_result.train_rows cannot be empty")
    if len(split_result.val_rows) == 0:
        raise ValueError(
            "split_result.val_rows cannot be empty; DatasetSplitter contract requires non-empty validation partition"
        )
    if len(split_result.test_rows) == 0:
        raise ValueError(
            "split_result.test_rows cannot be empty; DatasetSplitter contract requires non-empty test partition"
        )

    if b.last_train_observation_date != split_result.train_rows[-1].observation_date:
        raise ValueError(
            f"Boundary last_train_observation_date ({b.last_train_observation_date}) does not match "
            f"last retained train row ({split_result.train_rows[-1].observation_date})"
        )

    if b.first_val_observation_date != split_result.val_rows[0].observation_date:
        raise ValueError(
            f"Boundary first_val_observation_date ({b.first_val_observation_date}) does not match "
            f"first retained val row ({split_result.val_rows[0].observation_date})"
        )
    if b.last_val_observation_date != split_result.val_rows[-1].observation_date:
        raise ValueError(
            f"Boundary last_val_observation_date ({b.last_val_observation_date}) does not match "
            f"last retained val row ({split_result.val_rows[-1].observation_date})"
        )

    if b.first_test_observation_date != split_result.test_rows[0].observation_date:
        raise ValueError(
            f"Boundary first_test_observation_date ({b.first_test_observation_date}) does not match "
            f"first retained test row ({split_result.test_rows[0].observation_date})"
        )

    # 6. Verify raw partition allocations from SplitConfig and purge/embargo exclusion semantics
    cfg = split_result.config
    raw_train_size = math.floor(cfg.train_fraction * n)
    raw_val_size = math.floor(cfg.val_fraction * n)
    raw_test_size = n - raw_train_size - raw_val_size

    if raw_train_size < 1 or raw_val_size < 1 or raw_test_size < 1:
        raise ValueError(
            f"SplitConfig fractions ({cfg.train_fraction}, {cfg.val_fraction}, {cfg.test_fraction}) "
            f"allocate non-positive raw sizes ({raw_train_size}, {raw_val_size}, {raw_test_size}) "
            f"from {n} total rows"
        )

    raw_train = matrix.rows[:raw_train_size]
    raw_val = matrix.rows[raw_train_size : raw_train_size + raw_val_size]
    raw_test = matrix.rows[raw_train_size + raw_val_size :]

    # 6.1 Train partition must match raw_train exactly
    _verify_partition_rows_match("Train", split_result.train_rows, raw_train)

    # 6.2 Validation partition purge and embargo verification
    max_train_realization = max(r.realization_date for r in raw_train)
    expected_val_purged = 0
    if cfg.purge_overlap:
        for r in raw_val:
            if r.observation_date < max_train_realization:
                expected_val_purged += 1
            else:
                break
    if split_result.val_rows_purged != expected_val_purged:
        raise ValueError(
            f"Split purge/embargo mismatch for val: SplitResult val_rows_purged ({split_result.val_rows_purged}) "
            f"does not match expected purged count ({expected_val_purged})"
        )

    val_remaining_after_purge = raw_val_size - expected_val_purged
    expected_val_embargoed = 0
    if cfg.embargo_bars > 0 and val_remaining_after_purge > 0:
        expected_val_embargoed = min(cfg.embargo_bars, val_remaining_after_purge)
    if split_result.val_rows_embargoed != expected_val_embargoed:
        raise ValueError(
            f"Split purge/embargo mismatch for val: SplitResult val_rows_embargoed ({split_result.val_rows_embargoed}) "
            f"does not match expected embargoed count ({expected_val_embargoed})"
        )

    total_val_exclusions = expected_val_purged + expected_val_embargoed
    expected_val_rows = raw_val[total_val_exclusions:]
    _verify_partition_rows_match("val", split_result.val_rows, expected_val_rows)

    # 6.3 Test partition purge and embargo verification
    max_prior_realization = max(
        max_train_realization,
        max(r.realization_date for r in split_result.val_rows),
    )

    expected_test_purged = 0
    if cfg.purge_overlap:
        for r in raw_test:
            if r.observation_date < max_prior_realization:
                expected_test_purged += 1
            else:
                break
    if split_result.test_rows_purged != expected_test_purged:
        raise ValueError(
            f"Split purge/embargo mismatch for test: SplitResult test_rows_purged ({split_result.test_rows_purged}) "
            f"does not match expected purged count ({expected_test_purged})"
        )

    test_remaining_after_purge = raw_test_size - expected_test_purged
    expected_test_embargoed = 0
    if cfg.embargo_bars > 0 and test_remaining_after_purge > 0:
        expected_test_embargoed = min(cfg.embargo_bars, test_remaining_after_purge)
    if split_result.test_rows_embargoed != expected_test_embargoed:
        raise ValueError(
            f"Split purge/embargo mismatch for test: SplitResult test_rows_embargoed ({split_result.test_rows_embargoed}) "
            f"does not match expected embargoed count ({expected_test_embargoed})"
        )

    total_test_exclusions = expected_test_purged + expected_test_embargoed
    expected_test_rows = raw_test[total_test_exclusions:]
    _verify_partition_rows_match("test", split_result.test_rows, expected_test_rows)

    # 7. Validate row schema for every retained row
    num_features = len(matrix.feature_names)
    for split_label, partition_rows in (
        ("train", split_result.train_rows),
        ("val", split_result.val_rows),
        ("test", split_result.test_rows),
    ):
        for r_idx, r in enumerate(partition_rows):
            if len(r.feature_values) != num_features:
                raise ValueError(
                    f"{split_label} row {r_idx} feature width ({len(r.feature_values)}) "
                    f"does not match matrix feature count ({num_features})"
                )

    X_train, y_train = _rows_to_numpy_arrays(split_result.train_rows, num_features, "train")
    X_val, y_val = _rows_to_numpy_arrays(split_result.val_rows, num_features, "val")
    X_test, y_test = _rows_to_numpy_arrays(split_result.test_rows, num_features, "test")

    return NumpySplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=matrix.feature_names,
        feature_definitions=matrix.feature_definitions,
        split_result=split_result,
    )


def to_numpy_inference(
    inference_matrix: InferenceMatrix,
) -> NumpyInferenceDataset:
    """Convert an InferenceMatrix into a read-only float64 feature array.

    Verifies schema integrity against feature definitions.
    """
    if not isinstance(inference_matrix, InferenceMatrix):
        raise TypeError(
            f"inference_matrix must be an InferenceMatrix, got {type(inference_matrix).__name__}"
        )

    for idx, (name, defn) in enumerate(
        zip(inference_matrix.feature_names, inference_matrix.feature_definitions, strict=True)
    ):
        expected_name = generate_feature_name(defn)
        if name != expected_name:
            raise ValueError(
                f"InferenceMatrix feature_name at index {idx} ('{name}') does not match "
                f"expected generated name ('{expected_name}')"
            )

    n_rows = len(inference_matrix.rows)
    num_features = len(inference_matrix.feature_names)

    if n_rows == 0:
        X = np.empty((0, num_features), dtype=np.float64, order="C")
        X.flags.writeable = False
        return NumpyInferenceDataset(
            X=X,
            feature_names=inference_matrix.feature_names,
            feature_definitions=inference_matrix.feature_definitions,
            source_matrix=inference_matrix,
        )

    X_buf = np.empty((n_rows, num_features), dtype=np.float64, order="C")
    for i, row in enumerate(inference_matrix.rows):
        if len(row.feature_values) != num_features:
            raise ValueError(
                f"Inference row {i} width ({len(row.feature_values)}) does not match expected {num_features}"
            )
        for j, fv in enumerate(row.feature_values):
            X_buf[i, j] = _safe_decimal_to_float64(fv, f"inference feature [{i}, {j}]")

    X_buf.flags.writeable = False
    return NumpyInferenceDataset(
        X=X_buf,
        feature_names=inference_matrix.feature_names,
        feature_definitions=inference_matrix.feature_definitions,
        source_matrix=inference_matrix,
    )


def verify_schema_alignment(
    training: NumpySplit | FeatureMatrix,
    inference: NumpyInferenceDataset | InferenceMatrix,
) -> None:
    """Verify that training and inference datasets share an identical feature schema.

    Raises ValueError if column counts, column names, or column definitions differ or are reordered.
    """
    train_names = training.feature_names
    inf_names = inference.feature_names

    if len(train_names) != len(inf_names):
        raise ValueError(
            f"Feature count mismatch: training has {len(train_names)} columns, "
            f"inference has {len(inf_names)} columns"
        )

    if train_names != inf_names:
        raise ValueError(
            f"Feature column mismatch between training and inference: "
            f"training={train_names}, inference={inf_names}"
        )

    train_defs = training.feature_definitions
    inf_defs = inference.feature_definitions

    if train_defs != inf_defs:
        raise ValueError(
            f"FeatureDefinition mismatch between training and inference: "
            f"training={train_defs}, inference={inf_defs}"
        )
